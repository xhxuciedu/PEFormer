"""Final held-out evaluation of a heterogeneous, multi-architecture ensemble (round-3 §27).

Handles members with different architectures in one ensemble -- some with the Family C
feature branch, some without, some with layerwise context conditioning -- by reading each
checkpoint's own saved `model_config` rather than assuming a single shared architecture.

Each member is itself a 5-checkpoint set (one per official fold); a member's held-out
prediction is the mean over its 5 checkpoints, matching how OptiPrime's own released
5-checkpoint ensemble is evaluated. Members are then combined by rank-average, the rule
selected on matched dev folds (see reports/round3_research_log.md -- rank-average beat
mean-average on every dev fold, and equal weights beat fitted weights).

Requires --allow-heldout-evaluation; every run is logged by heldout_guard.

Member spec format (repeatable):
    --member NAME:GLOB[:feat]
e.g.
    --member familyC:'checkpoints/r2_familyC_*/best.pt':feat
    --member dapt:'checkpoints/r3_dapt_lr3e5_cv*/best.pt'
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.data.family_c_features import FEATURE_COLS, attach_family_c_features  # noqa: E402
from pe_rankformer.evaluation.heldout_guard import require_heldout_permission  # noqa: E402
from pe_rankformer.evaluation.metrics import global_metrics  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_heterogeneous_heldout")


@torch.no_grad()
def predict_one(ckpt_path: str, corpus, indices, device, batch_size=1024) -> np.ndarray:
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
    ap.add_argument("--member", action="append", required=True,
                    help="NAME:GLOB[:feat] -- repeatable; :feat marks a feature-branch member")
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/round3/heldout"))
    ap.add_argument("--corpus", default="data/processed/featurized_official.npz")
    ap.add_argument("--vocab", default="data/processed/context_vocab_official.json")
    ap.add_argument("--full-df", default="data/processed/optiprime_official_318471.parquet")
    ap.add_argument("--features-path", default="data/processed/family_c_features.parquet")
    ap.add_argument("--allow-heldout-evaluation", action="store_true")
    args = ap.parse_args()

    require_heldout_permission(
        args.allow_heldout_evaluation, script="evaluate_heterogeneous_heldout.py",
        reason=f"round-3 final ensemble eval, model-name={args.model_name}", n_rows=-1,
    )

    device = torch.device("cuda")
    vocab = ContextVocab.load(args.vocab)
    corpus_plain = load_featurized(args.corpus, vocab)
    full_df = pd.read_parquet(args.full_df)
    test_idx = np.where(corpus_plain.fold == 0)[0]
    logger.info("held-out rows: %d", len(test_idx))

    # Feature-branch members need the feature columns attached; normalization basis is
    # all of folds 1-5 (never the held-out fold), matching round-2's convention.
    train_idx_full = np.where(corpus_plain.fold != 0)[0]
    corpus_feat = attach_family_c_features(
        load_featurized(args.corpus, vocab), args.features_path, train_idx_full
    )

    member_preds = {}
    for spec in args.member:
        parts = spec.split(":")
        name, pattern = parts[0], parts[1]
        uses_feat = len(parts) > 2 and parts[2] == "feat"
        ckpts = sorted(globmod.glob(pattern))
        if not ckpts:
            raise SystemExit(f"member {name!r}: no checkpoints matched {pattern!r}")
        corpus = corpus_feat if uses_feat else corpus_plain
        logger.info("member %s: %d checkpoints, feature_branch=%s", name, len(ckpts), uses_feat)

        sub = []
        for ck in ckpts:
            p = predict_one(ck, corpus, test_idx, device)
            logger.info("    %-52s spearman=%.4f", Path(ck).parent.name,
                        global_metrics(corpus.target[test_idx], p).spearman)
            sub.append(p)
        member_preds[name] = np.mean(sub, axis=0)
        gm = global_metrics(corpus.target[test_idx], member_preds[name])
        logger.info("  MEMBER %s ensemble: spearman=%.4f pearson=%.4f", name, gm.spearman, gm.pearson)

    # Rank-average across members (rule selected on matched dev folds).
    ranks = [rankdata(p) / len(p) for p in member_preds.values()]
    preds = np.mean(ranks, axis=0)
    true = corpus_plain.target[test_idx]

    out = pd.DataFrame(
        {
            "record_id": full_df.iloc[test_idx]["record_id"].to_numpy(),
            "source_study": full_df.iloc[test_idx]["source_study"].to_numpy(),
            "cell_type": full_df.iloc[test_idx]["cell_type"].to_numpy(),
            "pe_type": full_df.iloc[test_idx]["pe_type"].to_numpy(),
            "true_efficiency": true,
            "predicted_efficiency": preds,
            "model": args.model_name,
        }
    )
    for name, p in member_preds.items():
        out[f"member_{name}"] = p

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_dir / f"predictions_{args.model_name}.parquet", index=False)

    # NOTE: rank-averaging changes the prediction SCALE (ranks in [0,1]), so MAE/RMSE
    # against raw efficiency are not meaningful for the blend and are reported only for
    # the individual members, where the scale is genuine predicted efficiency.
    gm = global_metrics(true, preds)
    logger.info("=" * 70)
    logger.info("ENSEMBLE (rank-avg) FULL HELD-OUT: spearman=%.4f pearson=%.4f (n=%d)",
                gm.spearman, gm.pearson, gm.n)
    by_study = {}
    for study, label in [("hsu2026", "Liu"), ("deepprime", "Kim")]:
        sub = out[out.source_study == study]
        g = global_metrics(sub.true_efficiency, sub.predicted_efficiency)
        by_study[label] = g.as_dict()
        logger.info("  %-4s (n=%5d): spearman=%.4f pearson=%.4f", label, len(sub), g.spearman, g.pearson)

    per_condition = []
    for (src, cell, pe), g in out.groupby(["source_study", "cell_type", "pe_type"]):
        if len(g) < 50:
            continue
        per_condition.append({
            "source": src, "cell_type": cell, "pe_type": pe, "n": len(g),
            "spearman": global_metrics(g.true_efficiency, g.predicted_efficiency).spearman,
        })

    metrics = {
        "model": args.model_name,
        "members": {k: len(globmod.glob(s.split(":")[1])) for k, s in zip(member_preds, args.member)},
        "n_test_rows": int(len(out)),
        "global": gm.as_dict(),
        "by_study": by_study,
        "per_condition": per_condition,
        "member_only_global": {
            k: global_metrics(true, v).as_dict() for k, v in member_preds.items()
        },
    }
    (args.out_dir / f"metrics_{args.model_name}.json").write_text(json.dumps(metrics, indent=2))
    logger.info("wrote %s", args.out_dir / f"metrics_{args.model_name}.json")


if __name__ == "__main__":
    main()
