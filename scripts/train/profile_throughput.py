"""GPU throughput/memory profiling across batch sizes (task spec §23-24).

Times several hundred steps after discarding warm-up, per the spec's instruction to
measure rather than assume training throughput.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402
from pe_rankformer.training.losses import LossWeights, total_loss  # noqa: E402
from pe_rankformer.training.ranking import GroupedBatchSampler, sample_ranking_pairs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("profile_throughput")

WARMUP_STEPS = 20
TIMED_STEPS = 100


def profile_batch_size(model, corpus, batch_size, device, weights):
    sampler = GroupedBatchSampler(corpus.group_key, batch_size=batch_size, seed=0, drop_last=True)
    batches = iter(sampler)
    ds = PEDataset(corpus)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler_dtype = torch.bfloat16

    def make_batch():
        idx = next(batches)
        b = collate([ds[i] for i in idx])
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    for _ in range(WARMUP_STEPS):
        b = make_batch()
        with torch.autocast(device_type="cuda", dtype=scaler_dtype):
            score = model(b)
            pi, pj = sample_ranking_pairs(b["group_key"], b["target"], min_diff=0.02, max_pairs_per_group=4)
            loss, _ = total_loss(score, b["target"], pi, pj, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        b = make_batch()
        with torch.autocast(device_type="cuda", dtype=scaler_dtype):
            score = model(b)
            pi, pj = sample_ranking_pairs(b["group_key"], b["target"], min_diff=0.02, max_pairs_per_group=4)
            loss, _ = total_loss(score, b["target"], pi, pj, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    step_time = elapsed / TIMED_STEPS
    examples_per_sec = batch_size / step_time
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return step_time, examples_per_sec, peak_mem_gb


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required for profiling"
    device = torch.device("cuda")
    logger.info("device: %s", torch.cuda.get_device_name(0))

    vocab = ContextVocab.load("data/processed/context_vocab.json")
    corpus = load_featurized("data/processed/featurized_corpus.npz", vocab)
    n_train = int(len(corpus) * 0.9)  # rough estimate for reporting; exact split done at train time
    logger.info("corpus rows: %d", len(corpus))

    cfg = PERankFormerConfig(context_fields=vocab.fields, context_vocab_sizes=vocab.sizes())
    weights = LossWeights(lambda_rank=0.25)

    rows = []
    for bs in (128, 256, 512, 1024):
        model = PERankFormer(cfg).to(device)
        try:
            step_time, ex_per_sec, peak_mem = profile_batch_size(model, corpus, bs, device, weights)
        except torch.cuda.OutOfMemoryError:
            logger.warning("batch_size=%d: OOM", bs)
            torch.cuda.empty_cache()
            continue
        steps_per_epoch = n_train // bs
        minutes_per_epoch = steps_per_epoch * step_time / 60
        logger.info(
            "batch=%4d  step_time=%.4fs  examples/sec=%.0f  peak_mem=%.2fGB  "
            "steps/epoch=%d  min/epoch=%.2f",
            bs, step_time, ex_per_sec, peak_mem, steps_per_epoch, minutes_per_epoch,
        )
        rows.append(
            dict(batch_size=bs, step_time_s=step_time, examples_per_sec=ex_per_sec,
                 peak_mem_gb=peak_mem, steps_per_epoch=steps_per_epoch,
                 minutes_per_epoch=minutes_per_epoch)
        )
        del model
        torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r["examples_per_sec"]) if rows else None

    report = Path("reports/compute_profile.md")
    lines = [
        "# Compute profile\n",
        f"GPU: `{torch.cuda.get_device_name(0)}` (device 6, CUDA_VISIBLE_DEVICES=6)\n",
        f"Model: PE-RankFormer, {PERankFormer(cfg).num_parameters()/1e6:.1f}M params, "
        f"d_model={cfg.d_model}\n",
        f"Sequence lengths: edit={cfg.edit_seq_len}, pegRNA={cfg.peg_seq_len}\n",
        f"Precision: BF16 autocast, AdamW, fused SDPA (PyTorch default backend)\n",
        f"Corpus: {len(corpus):,} rows ({n_train:,} assumed train split for epoch-time estimate)\n",
        "\n## Batch size scaling (100 timed steps after 20 warm-up)\n",
        "| batch | step time (s) | examples/sec | peak mem (GB) | steps/epoch | min/epoch |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['batch_size']} | {r['step_time_s']:.4f} | {r['examples_per_sec']:.0f} | "
            f"{r['peak_mem_gb']:.2f} | {r['steps_per_epoch']} | {r['minutes_per_epoch']:.2f} |"
        )
    if best:
        full_run_hours_20ep = best["minutes_per_epoch"] * 20 / 60
        full_run_hours_30ep = best["minutes_per_epoch"] * 30 / 60
        lines += [
            "",
            f"## Recommendation",
            f"Best throughput at batch_size={best['batch_size']} "
            f"({best['examples_per_sec']:.0f} examples/sec, {best['peak_mem_gb']:.2f} GB peak).",
            f"Estimated full training time: {full_run_hours_20ep:.2f}h (20 epochs) - "
            f"{full_run_hours_30ep:.2f}h (30 epochs), single model, single fold.",
        ]
    report.write_text("\n".join(lines))
    logger.info("wrote %s", report)


if __name__ == "__main__":
    main()
