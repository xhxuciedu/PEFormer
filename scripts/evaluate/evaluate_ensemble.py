"""Evaluate an ensemble of PE-RankFormer checkpoints on the locked test fold.

Motivation (fairness): the OptiPrime baseline we compare against is itself a 5-model
ensemble -- `PREDICT_PE.py` averages all 5 released fold checkpoints by default. Comparing
a single PE-RankFormer run against that 5-model average understates our side. This script
produces the like-for-like comparison by averaging N independently-seeded PE-RankFormer
runs of the same configuration.
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
logger = logging.getLogger("evaluate_ensemble")


@torch.no_grad()
def predict_one(ckpt_path: Path, corpus, indices, device, batch_size=1024) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PERankFormer(PERankFormerConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
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
    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/runs/eval_test_fold"))
    ap.add_argument("--corpus", default="data/processed/featurized_corpus.npz")
    ap.add_argument("--vocab", default="data/processed/context_vocab.json")
    ap.add_argument("--full-df", default="data/processed/optiprime_full_297962.parquet",
                    help="row-aligned parquet the featurized corpus was built from")
    ap.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Evaluate on the validation fold (for choosing ensemble composition) or the "
             "locked test fold (final report number only, requires --allow-heldout-evaluation).",
    )
    ap.add_argument(
        "--allow-heldout-evaluation", action="store_true",
        help="Required in addition to --split test. See heldout_guard.py.",
    )
    args = ap.parse_args()

    if args.split == "test":
        require_heldout_permission(
            args.allow_heldout_evaluation, script="evaluate_ensemble.py",
            reason=f"model-name={args.model_name}", n_rows=-1,
        )

    device = torch.device("cuda")
    vocab = ContextVocab.load(args.vocab)
    corpus = load_featurized(args.corpus, vocab)
    full_df = pd.read_parquet(args.full_df)

    first = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    fold_key = "test_fold" if args.split == "test" else "val_fold"
    test_fold = first["config"]["data"][fold_key]
    test_idx = np.where(corpus.fold == test_fold)[0]
    logger.info(
        "ensembling %d checkpoints over %d rows of the %s fold (%d)",
        len(args.checkpoints), len(test_idx), args.split, test_fold,
    )

    member_preds = []
    for ck in args.checkpoints:
        p = predict_one(ck, corpus, test_idx, device)
        gm = global_metrics(corpus.target[test_idx], p)
        logger.info("  member %s: spearman=%.4f", ck.parent.name, gm.spearman)
        member_preds.append(p)

    preds = np.mean(member_preds, axis=0)
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
            "fold": test_fold,
            "model": args.model_name,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_dir / f"predictions_{args.model_name}.parquet", index=False)

    gm = global_metrics(true, preds)
    logger.info("ENSEMBLE GLOBAL: pearson=%.4f spearman=%.4f mae=%.4f rmse=%.4f", gm.pearson, gm.spearman, gm.mae, gm.rmse)

    wt = within_target_spearman(out, "target_group", "true_efficiency", "predicted_efficiency", min_n=5)
    point, lo, hi = target_level_bootstrap_ci(wt.spearman.tolist())
    r1 = top_k_regret(out, "target_group", "true_efficiency", "predicted_efficiency", k=1)
    r3 = top_k_regret(out, "target_group", "true_efficiency", "predicted_efficiency", k=3)
    r1p, r1lo, r1hi = target_level_bootstrap_ci(r1.regret.tolist())
    r3p, r3lo, r3hi = target_level_bootstrap_ci(r3.regret.tolist())
    logger.info("WITHIN-TARGET: %.4f [%.4f, %.4f]", point, lo, hi)
    logger.info("TOP-1 REGRET: %.4f [%.4f, %.4f]  TOP-3: %.4f", r1p, r1lo, r1hi, r3p)

    metrics = {
        "model": args.model_name,
        "n_members": len(args.checkpoints),
        "members": [str(c) for c in args.checkpoints],
        "n_test_rows": int(len(out)),
        "global": gm.as_dict(),
        "within_target_spearman_macro_mean": point,
        "within_target_spearman_ci95": [lo, hi],
        "top1_regret": r1p,
        "top1_regret_ci95": [r1lo, r1hi],
        "top3_regret": r3p,
        "top3_regret_ci95": [r3lo, r3hi],
        "top1_recall": top_k_recall(out, "target_group", "true_efficiency", "predicted_efficiency", k=1),
        "top3_recall": top_k_recall(out, "target_group", "true_efficiency", "predicted_efficiency", k=3),
        "ndcg3": ndcg_at_k(out, "target_group", "true_efficiency", "predicted_efficiency", k=3),
        "ndcg5": ndcg_at_k(out, "target_group", "true_efficiency", "predicted_efficiency", k=5),
    }
    (args.out_dir / f"metrics_{args.model_name}.json").write_text(json.dumps(metrics, indent=2))
    logger.info("wrote %s", args.out_dir / f"metrics_{args.model_name}.json")


if __name__ == "__main__":
    main()
