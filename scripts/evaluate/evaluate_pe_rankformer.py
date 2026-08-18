"""Evaluate a trained PE-RankFormer checkpoint on the locked test fold.

Produces per-row predictions (task spec §42) and the primary metric tables
(task spec §30-33).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.evaluation.heldout_guard import require_heldout_permission  # noqa: E402
from pe_rankformer.evaluation.metrics import (  # noqa: E402
    global_metrics,
    ndcg_at_k,
    target_level_bootstrap_ci,
    top_k_recall,
    top_k_regret,
    within_target_spearman,
)
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_pe_rankformer")


@torch.no_grad()
def predict(model, corpus, indices, device, batch_size=1024):
    model.eval()
    ds = PEDataset(corpus)
    preds = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        batch = collate([ds[i] for i in idx])
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        preds.append(model.efficiency_from_output(out).float().cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--allow-heldout-evaluation", action="store_true",
        help="Required: this script always evaluates on the locked test fold. See heldout_guard.py.",
    )
    args = ap.parse_args()

    require_heldout_permission(
        args.allow_heldout_evaluation, script="evaluate_pe_rankformer.py",
        reason=f"model-name={args.model_name} checkpoint={args.checkpoint}", n_rows=-1,
    )

    device = torch.device("cuda")
    vocab = ContextVocab.load("data/processed/context_vocab.json")
    corpus = load_featurized("data/processed/featurized_corpus.npz", vocab)
    full_df = pd.read_parquet("data/processed/optiprime_full_297962.parquet")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = PERankFormerConfig(**ckpt["model_config"])
    model = PERankFormer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("loaded checkpoint from epoch %d", ckpt["epoch"])

    test_idx = np.where(corpus.fold == ckpt["config"]["data"]["test_fold"])[0]
    logger.info("test fold rows: %d (locked, evaluated once)", len(test_idx))

    preds = predict(model, corpus, test_idx, device)
    true = corpus.target[test_idx]

    out = pd.DataFrame(
        {
            "record_id": full_df.iloc[test_idx]["record_id"].to_numpy(),
            "target_group": corpus.group_key[test_idx],
            "group": full_df.iloc[test_idx]["group"].to_numpy(),
            "source_study": full_df.iloc[test_idx]["source_study"].to_numpy(),
            "cell_type": full_df.iloc[test_idx]["cell_type"].to_numpy(),
            "pe_type": full_df.iloc[test_idx]["pe_type"].to_numpy(),
            "true_efficiency": true,
            "predicted_efficiency": preds,
            "fold": ckpt["config"]["data"]["test_fold"],
            "model": args.model_name,
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_dir / f"predictions_{args.model_name}.parquet", index=False)

    gm = global_metrics(out.true_efficiency, out.predicted_efficiency)
    logger.info("GLOBAL: pearson=%.4f spearman=%.4f mae=%.4f rmse=%.4f (n=%d)", gm.pearson, gm.spearman, gm.mae, gm.rmse, gm.n)

    wt = within_target_spearman(out, "target_group", "true_efficiency", "predicted_efficiency", min_n=5)
    point, lo, hi = target_level_bootstrap_ci(wt.spearman.tolist())
    logger.info(
        "WITHIN-TARGET SPEARMAN (n>=5 groups, %d groups): macro_mean=%.4f [%.4f, %.4f]  median=%.4f",
        len(wt), point, lo, hi, wt.spearman.median() if len(wt) else float("nan"),
    )

    r1 = top_k_regret(out, "target_group", "true_efficiency", "predicted_efficiency", k=1)
    r3 = top_k_regret(out, "target_group", "true_efficiency", "predicted_efficiency", k=3)
    recall1 = top_k_recall(out, "target_group", "true_efficiency", "predicted_efficiency", k=1)
    recall3 = top_k_recall(out, "target_group", "true_efficiency", "predicted_efficiency", k=3)
    recall5 = top_k_recall(out, "target_group", "true_efficiency", "predicted_efficiency", k=5)
    ndcg3 = ndcg_at_k(out, "target_group", "true_efficiency", "predicted_efficiency", k=3)
    ndcg5 = ndcg_at_k(out, "target_group", "true_efficiency", "predicted_efficiency", k=5)

    r1_point, r1_lo, r1_hi = target_level_bootstrap_ci(r1.regret.tolist())
    r3_point, r3_lo, r3_hi = target_level_bootstrap_ci(r3.regret.tolist())

    logger.info("TOP-1 REGRET: %.4f [%.4f, %.4f] (n=%d groups)", r1_point, r1_lo, r1_hi, len(r1))
    logger.info("TOP-3 REGRET: %.4f [%.4f, %.4f] (n=%d groups)", r3_point, r3_lo, r3_hi, len(r3))
    logger.info("TOP-1/3/5 RECALL: %.4f / %.4f / %.4f", recall1, recall3, recall5)
    logger.info("NDCG@3/5: %.4f / %.4f", ndcg3, ndcg5)

    metrics = {
        "model": args.model_name,
        "n_test_rows": int(len(out)),
        "n_target_groups": int(out.target_group.nunique()),
        "global": gm.as_dict(),
        "within_target_spearman_macro_mean": point,
        "within_target_spearman_ci95": [lo, hi],
        "within_target_spearman_median": float(wt.spearman.median()) if len(wt) else None,
        "n_groups_within_target_eval": int(len(wt)),
        "top1_regret": r1_point,
        "top1_regret_ci95": [r1_lo, r1_hi],
        "top3_regret": r3_point,
        "top3_regret_ci95": [r3_lo, r3_hi],
        "top1_recall": recall1,
        "top3_recall": recall3,
        "top5_recall": recall5,
        "ndcg3": ndcg3,
        "ndcg5": ndcg5,
    }
    (args.out_dir / f"metrics_{args.model_name}.json").write_text(json.dumps(metrics, indent=2))
    logger.info("wrote %s", args.out_dir / f"metrics_{args.model_name}.json")


if __name__ == "__main__":
    main()
