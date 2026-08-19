"""Evaluate one or more checkpoints (ensembled if >1) on the round-3 Liu+Kim-matched
development folds (spec §6-7). Never touches the official held-out set.

Handles both feature-branch models (round-2 Family C/D) and plain models
(round-1 baseline, round-2 Family A) via --feature-branch.

For each dev-fold column in --dev-folds-file (default: all three
round3_dev_fold_{0,1,2}), evaluates on that fold's "val" rows and reports:
  - combined Liu+Kim Spearman (primary selection metric, spec §7)
  - Liu-only and Kim-only Spearman
  - macro-context Spearman (mean Spearman over cell_type x pe_type groups, n>=20)
  - Pearson, MAE, RMSE

Reports a per-fold table plus the mean across folds.
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
from pe_rankformer.data.family_c_features import FEATURE_COLS, attach_family_c_features  # noqa: E402
from pe_rankformer.evaluation.metrics import global_metrics  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_on_devfolds")


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


def macro_context_spearman(df: pd.DataFrame, min_n: int = 20) -> float:
    from scipy.stats import spearmanr

    rhos = []
    for _, g in df.groupby(["cell_type", "pe_type"]):
        if len(g) < min_n:
            continue
        rhos.append(spearmanr(g.true_efficiency, g.predicted_efficiency).statistic)
    return float(np.mean(rhos)) if rhos else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/round3/dev_recalibration"))
    ap.add_argument("--corpus", default="data/processed/featurized_official.npz")
    ap.add_argument("--vocab", default="data/processed/context_vocab_official.json")
    ap.add_argument("--full-df", default="data/processed/optiprime_official_318471.parquet")
    ap.add_argument("--dev-folds-file", default="data/processed/round3_dev_assignments.parquet")
    ap.add_argument("--dev-fold-cols", nargs="+", default=None,
                    help="default: all round3_dev_fold_{0,1,2} columns found")
    ap.add_argument("--feature-branch", action="store_true")
    ap.add_argument("--features-path", default="data/processed/family_c_features.parquet")
    ap.add_argument(
        "--oof", action="store_true",
        help="Out-of-fold prediction: score each row with ONLY the checkpoint that held "
             "that row's official fold out of training. Required for an unbiased read on "
             "checkpoints trained across the official 5-fold split, since the round-3 dev "
             "folds are drawn from those same folds 1-5 -- a plain 5-model ensemble average "
             "is ~80%% in-sample on every dev row (4 of 5 members trained on it).",
    )
    args = ap.parse_args()

    device = torch.device("cuda")
    vocab = ContextVocab.load(args.vocab)
    corpus = load_featurized(args.corpus, vocab)
    full_df = pd.read_parquet(args.full_df)
    dev = pd.read_parquet(args.dev_folds_file)

    dev_cols = args.dev_fold_cols or [c for c in dev.columns if c.startswith("round3_dev_fold_")]
    logger.info("evaluating %s on dev folds: %s", args.model_name, dev_cols)

    record_id_to_pos = {rid: i for i, rid in enumerate(corpus.record_id)}

    ckpt_val_fold = {}
    if args.oof:
        for ck in args.checkpoints:
            c = torch.load(ck, map_location="cpu", weights_only=False)
            ckpt_val_fold[ck] = c["config"]["data"]["val_fold"]
        covered = sorted(ckpt_val_fold.values())
        logger.info("OOF mode: checkpoint val_folds = %s", covered)
        if sorted(set(covered)) != [1, 2, 3, 4, 5]:
            raise SystemExit(
                f"OOF mode needs exactly one checkpoint per official fold 1-5, got val_folds {covered}"
            )

    if args.feature_branch:
        # Normalize once using the full training pool (folds 1-5), same convention as
        # the round-2 held-out evaluation -- a single principled basis independent of
        # which dev fold is being scored.
        train_idx_full = np.where(corpus.fold != 0)[0]
        corpus = attach_family_c_features(corpus, args.features_path, train_idx_full)
        logger.info("attached %d Family-C features (normalized on %d rows)", len(FEATURE_COLS), len(train_idx_full))

    per_fold_rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for col in dev_cols:
        sub = dev[["record_id", col]].dropna()
        val_ids = sub.loc[sub[col] == "val", "record_id"]
        val_idx = np.array([record_id_to_pos[r] for r in val_ids], dtype=np.int64)

        if args.oof:
            # Each row scored only by the checkpoint that held its official fold out.
            preds = np.full(len(val_idx), np.nan, dtype=np.float64)
            row_official_fold = corpus.fold[val_idx]
            for ck, vf in ckpt_val_fold.items():
                mask = row_official_fold == vf
                if not mask.any():
                    continue
                preds[mask] = predict_one(ck, corpus, val_idx[mask], device)
            assert not np.isnan(preds).any(), "some dev rows had no out-of-fold checkpoint"
        else:
            member_preds = [predict_one(ck, corpus, val_idx, device) for ck in args.checkpoints]
            preds = np.mean(member_preds, axis=0)
        true = corpus.target[val_idx]

        out = pd.DataFrame(
            {
                "record_id": full_df.iloc[val_idx]["record_id"].to_numpy(),
                "source_study": full_df.iloc[val_idx]["source_study"].to_numpy(),
                "cell_type": full_df.iloc[val_idx]["cell_type"].to_numpy(),
                "pe_type": full_df.iloc[val_idx]["pe_type"].to_numpy(),
                "true_efficiency": true,
                "predicted_efficiency": preds,
            }
        )
        gm = global_metrics(true, preds)
        liu = out[out.source_study == "hsu2026"]
        kim = out[out.source_study == "deepprime"]
        gm_liu = global_metrics(liu.true_efficiency, liu.predicted_efficiency)
        gm_kim = global_metrics(kim.true_efficiency, kim.predicted_efficiency)
        macro = macro_context_spearman(out)

        out["dev_fold"] = col
        out.to_parquet(args.out_dir / f"predictions_{args.model_name}_{col}.parquet", index=False)

        row = {
            "dev_fold": col, "n": len(out),
            "spearman": gm.spearman, "pearson": gm.pearson, "mae": gm.mae, "rmse": gm.rmse,
            "liu_spearman": gm_liu.spearman, "liu_n": len(liu),
            "kim_spearman": gm_kim.spearman, "kim_n": len(kim),
            "macro_context_spearman": macro,
        }
        per_fold_rows.append(row)
        logger.info(
            "%s: n=%d combined=%.4f liu=%.4f kim=%.4f macro_ctx=%.4f",
            col, row["n"], row["spearman"], row["liu_spearman"], row["kim_spearman"], row["macro_context_spearman"],
        )

    table = pd.DataFrame(per_fold_rows)
    summary = {
        "model": args.model_name,
        "checkpoints": [str(c) for c in args.checkpoints],
        "per_fold": per_fold_rows,
        "mean_spearman": float(table.spearman.mean()),
        "mean_liu_spearman": float(table.liu_spearman.mean()),
        "mean_kim_spearman": float(table.kim_spearman.mean()),
        "mean_macro_context_spearman": float(table.macro_context_spearman.mean()),
        "mean_pearson": float(table.pearson.mean()),
        "mean_mae": float(table.mae.mean()),
    }
    logger.info(
        "MEAN across %d dev folds: combined=%.4f liu=%.4f kim=%.4f",
        len(table), summary["mean_spearman"], summary["mean_liu_spearman"], summary["mean_kim_spearman"],
    )
    (args.out_dir / f"{args.model_name}.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", args.out_dir / f"{args.model_name}.json")


if __name__ == "__main__":
    main()
