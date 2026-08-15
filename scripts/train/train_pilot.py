"""Train PE-RankFormer on the locked protospacer-disjoint split (task spec §22, §26-27).

Usage:
    python scripts/train/train_pilot.py --run-name model_a --lambda-rank 0.25
    python scripts/train/train_pilot.py --run-name model_b_norank --lambda-rank 0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.evaluation.metrics import global_metrics  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402
from pe_rankformer.training.losses import LossWeights, total_loss  # noqa: E402
from pe_rankformer.training.ranking import GroupedBatchSampler, sample_ranking_pairs  # noqa: E402
from pe_rankformer.training.schedule import warmup_cosine_schedule  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_pilot")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, corpus, indices, device, batch_size=1024):
    model.eval()
    ds = PEDataset(corpus)
    preds, targets = [], []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        batch = collate([ds[i] for i in idx])
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            score = model(batch)
        preds.append(torch.sigmoid(score).float().cpu().numpy())
        targets.append(batch["target"].float().cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return global_metrics(targets, preds), preds, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--lambda-rank", type=float, default=None, help="override loss.lambda_rank")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument(
        "--no-context", action="store_true", help="Model C ablation: disable FiLM context conditioning"
    )
    ap.add_argument("--patience", type=int, default=None, help="override early-stop patience")
    ap.add_argument(
        "--regression-space", choices=["raw", "logit"], default=None,
        help="regression loss space (task spec §19 comparison)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.lambda_rank is not None:
        cfg["loss"]["lambda_rank"] = args.lambda_rank
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.no_context:
        cfg["model"]["use_context"] = False
    if args.patience is not None:
        cfg["train"]["early_stop_patience"] = args.patience
    if args.regression_space is not None:
        cfg["loss"]["regression_space"] = args.regression_space

    set_seed(cfg["seed"])
    device = torch.device("cuda")
    logger.info("device: %s", torch.cuda.get_device_name(0))

    vocab = ContextVocab.load(cfg["data"]["vocab"])
    corpus_path = Path(cfg["data"]["corpus"])
    corpus = load_featurized(str(corpus_path), vocab)
    dataset_hash = file_sha256(corpus_path)
    logger.info("corpus: %d rows, hash=%s", len(corpus), dataset_hash[:12])

    fold = corpus.fold
    test_idx = np.where(fold == cfg["data"]["test_fold"])[0]
    val_idx = np.where(fold == cfg["data"]["val_fold"])[0]
    train_idx = np.where(np.isin(fold, cfg["data"]["train_folds"]))[0]
    logger.info("train=%d val=%d test=%d (test fold locked, not touched)", len(train_idx), len(val_idx), len(test_idx))

    model_cfg = PERankFormerConfig(
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), **cfg["model"]
    )
    model = PERankFormer(model_cfg).to(device)
    logger.info("model: %.1fM params", model.num_parameters() / 1e6)

    weights = LossWeights(
        lambda_rank=cfg["loss"]["lambda_rank"],
        huber_beta=cfg["loss"]["huber_beta"],
        min_pair_diff=cfg["loss"]["min_pair_diff"],
        regression_space=cfg["loss"].get("regression_space", "raw"),
    )
    max_pairs_per_group = cfg["loss"]["max_pairs_per_group"]

    batch_size = cfg["train"]["batch_size"]
    sampler = GroupedBatchSampler(corpus.group_key[train_idx], batch_size=batch_size, seed=cfg["seed"])
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * cfg["train"]["max_epochs"]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    sched = warmup_cosine_schedule(opt, total_steps, warmup_frac=cfg["optim"]["warmup_frac"])

    run_id = f"{args.run_name}_{int(time.time())}"
    run_dir = Path("results/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))

    history_rows = []
    best_val_spearman = -1.0
    best_epoch = -1
    patience_counter = 0
    ds = PEDataset(corpus)

    start_time = time.time()
    for epoch in range(cfg["train"]["max_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        epoch_t0 = time.time()
        running = {"loss": 0.0, "reg": 0.0, "rank": 0.0, "n_pairs": 0}
        n_batches = 0
        for local_batch in sampler:
            global_idx = train_idx[local_batch]
            batch = collate([ds[i] for i in global_idx])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                score = model(batch)
                pi, pj = sample_ranking_pairs(
                    batch["group_key"], batch["target"],
                    min_diff=cfg["loss"]["min_pair_diff"], max_pairs_per_group=max_pairs_per_group,
                )
                loss, parts = total_loss(score, batch["target"], pi, pj, weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            sched.step()

            for k in ("loss", "reg", "rank"):
                running[k] += parts[k]
            running["n_pairs"] += pi.numel()
            n_batches += 1

        epoch_time = time.time() - epoch_t0
        val_metrics, _, _ = evaluate(model, corpus, val_idx, device)
        lr_now = sched.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "train_loss": running["loss"] / n_batches,
            "train_reg_loss": running["reg"] / n_batches,
            "train_rank_loss": running["rank"] / n_batches,
            "train_pairs_per_epoch": running["n_pairs"],
            "val_pearson": val_metrics.pearson,
            "val_spearman": val_metrics.spearman,
            "val_mae": val_metrics.mae,
            "val_rmse": val_metrics.rmse,
            "lr": lr_now,
            "epoch_time_s": epoch_time,
        }
        history_rows.append(row)
        logger.info(
            "epoch %2d/%d  train_loss=%.4f (reg=%.4f rank=%.4f)  val_spearman=%.4f val_pearson=%.4f "
            "val_mae=%.4f  time=%.1fs",
            epoch, cfg["train"]["max_epochs"] - 1, row["train_loss"], row["train_reg_loss"],
            row["train_rank_loss"], val_metrics.spearman, val_metrics.pearson, val_metrics.mae, epoch_time,
        )

        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)

        improved = val_metrics.spearman > best_val_spearman
        if improved:
            best_val_spearman = val_metrics.spearman
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "model_config": model_cfg.__dict__, "epoch": epoch},
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1

        if epoch >= cfg["train"]["early_stop_min_warmup_epochs"] and patience_counter >= cfg["train"]["early_stop_patience"]:
            logger.info("early stopping at epoch %d (best epoch %d, val_spearman=%.4f)", epoch, best_epoch, best_val_spearman)
            break

    total_time = time.time() - start_time
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg, "model_config": model_cfg.__dict__, "epoch": epoch},
        ckpt_dir / "final.pt",
    )

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_info = {
        "run_id": run_id,
        "run_name": args.run_name,
        "lambda_rank": cfg["loss"]["lambda_rank"],
        "seed": cfg["seed"],
        "git_commit": git_commit,
        "dataset_hash": dataset_hash,
        "gpu": torch.cuda.get_device_name(0),
        "n_params": model.num_parameters(),
        "best_epoch": best_epoch,
        "best_val_spearman": best_val_spearman,
        "total_epochs_run": epoch + 1,
        "total_train_time_s": total_time,
        "checkpoint_best": str(ckpt_dir / "best.pt"),
        "checkpoint_final": str(ckpt_dir / "final.pt"),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    logger.info("done: %s", json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
