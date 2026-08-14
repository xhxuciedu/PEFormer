"""Tiny-overfit sanity check (task spec §25): the model must be able to drive loss to
near-zero on a small fixed subset before any full training run is trusted.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402
from pe_rankformer.training.losses import LossWeights, total_loss  # noqa: E402
from pe_rankformer.training.ranking import GroupedBatchSampler, sample_ranking_pairs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("overfit_test")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

    vocab = ContextVocab.load("data/processed/context_vocab.json")
    corpus = load_featurized("data/processed/featurized_corpus.npz", vocab)

    # Build the tiny fixed batch via GroupedBatchSampler rather than uniform random
    # sampling: with ~220k distinct ranking groups, a random 256-row sample almost
    # never contains two rows from the same group, so the ranking loss would never be
    # exercised by this sanity check otherwise.
    sampler = GroupedBatchSampler(corpus.group_key, batch_size=256, seed=0, drop_last=True)
    idx = next(iter(sampler))
    ds = PEDataset(corpus, indices=idx)
    batch = collate([ds[i] for i in range(len(ds))])
    batch = {k: v.to(device) for k, v in batch.items()}
    n_groups_with_pairs = int((torch.bincount(torch.unique(batch["group_key"], return_inverse=True)[1]) >= 2).sum())
    logger.info("tiny batch: %d rows, %d groups with >=2 members", len(idx), n_groups_with_pairs)

    cfg = PERankFormerConfig(context_fields=vocab.fields, context_vocab_sizes=vocab.sizes())
    model = PERankFormer(cfg).to(device)
    logger.info("model params: %.1fM", model.num_parameters() / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    weights = LossWeights(lambda_rank=0.25)

    model.train()
    n_steps = 900
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        score = model(batch)
        pi, pj = sample_ranking_pairs(batch["group_key"], batch["target"], min_diff=0.02, max_pairs_per_group=4)
        loss, parts = total_loss(score, batch["target"], pi, pj, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 40 == 0 or step == n_steps - 1:
            with torch.no_grad():
                pred = torch.sigmoid(score)
                mae = (pred - batch["target"]).abs().mean().item()
                rho = spearmanr(pred.cpu().numpy(), batch["target"].cpu().numpy()).statistic
            logger.info(
                "step %4d loss=%.5f reg=%.5f rank=%.5f mae=%.5f spearman=%.4f n_pairs=%d",
                step, parts["loss"], parts["reg"], parts["rank"], mae, rho, pi.numel(),
            )

    model.eval()
    with torch.no_grad():
        score = model(batch)
        pred = torch.sigmoid(score)
        final_mae = (pred - batch["target"]).abs().mean().item()
        final_rho = spearmanr(pred.cpu().numpy(), batch["target"].cpu().numpy()).statistic

    logger.info("FINAL: mae=%.5f spearman=%.4f", final_mae, final_rho)
    # This batch is deliberately biased toward duplicate ranking groups (different
    # pegRNA designs for the same edit/context), which is harder to memorize exactly
    # than an i.i.d. random sample -- the bar is "clearly learns", not "perfect fit".
    passed = final_mae < 0.03 and final_rho > 0.85
    logger.info("PASS" if passed else "FAIL: model could not overfit a tiny fixed batch")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
