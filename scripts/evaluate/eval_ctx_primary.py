"""Evaluate a ctx-primary model: invert the within-condition quantile back to efficiency.

Round-7 diagnosis: the model's rankings are far more similar across experimental
conditions than reality is (0.835 vs 0.683), i.e. it leans on the cross-condition mean
shift instead of learning how designs reorder. Layerwise FiLM made that *worse*
(+0.1923 excess), because FiLM is structurally a scale-and-shift and more of it
strengthens the very pathway at fault.

The ctx-primary model attacks the objective instead of the architecture. It is trained
on within-condition quantiles F_c(y), so the mean shift earns no credit and the
sequence model must learn ranking. Location is then supplied *exactly*, by mapping the
predicted quantile back through each condition's empirical training CDF:

    yhat = F_c^{-1}( qhat(sequence, context) )

That inversion is what makes predictions comparable across conditions again. Without
it the raw quantiles are all on [0,1] regardless of condition, and pooled Spearman
would be measuring nothing.

The CDFs are the ones stored in the checkpoint at training time, so they are fitted on
training rows only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_ctx_primary")


@torch.no_grad()
def predict(ckpt_path: str, corpus, idx, device, batch_size=1024) -> tuple[np.ndarray, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    ds = PEDataset(corpus)
    out = []
    for s in range(0, len(idx), batch_size):
        b = collate([ds[i] for i in idx[s:s + batch_size]])
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = model(b)
        out.append(model.efficiency_from_output(o).float().cpu().numpy())
    return np.concatenate(out), ck["config"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-fold-col", default="round3_dev_fold_0")
    ap.add_argument("--corpus", default="data/processed/featurized_official.npz")
    ap.add_argument("--vocab", default="data/processed/context_vocab_official.json")
    ap.add_argument("--full-df", default="data/processed/optiprime_official_318471.parquet")
    ap.add_argument("--dev-folds-file", default="data/processed/round3_dev_assignments.parquet")
    args = ap.parse_args()

    vocab = ContextVocab.load(args.vocab)
    corpus = load_featurized(args.corpus, vocab)
    dev = pd.read_parquet(args.dev_folds_file, columns=["record_id", args.dev_fold_col]).dropna()
    pos = {r: i for i, r in enumerate(corpus.record_id)}
    sub = dev[dev[args.dev_fold_col] == "val"]
    idx = np.array([pos[r] for r in sub.record_id if r in pos])

    device = torch.device("cuda")
    q, cfg = predict(args.checkpoint, corpus, idx, device)

    meta = pd.read_parquet(args.full_df, columns=["record_id", "source_study", "cell_type", "pe_type"])
    df = pd.DataFrame({"record_id": corpus.record_id[idx],
                       "true_efficiency": corpus.target[idx], "q": q}).merge(meta, on="record_id")

    cp = cfg.get("ctx_primary")
    assert cp is not None, "checkpoint was not trained with --ctx-primary"
    key = df[cp["fields"]].astype(str).agg("|".join, axis=1).to_numpy()

    # Invert per condition: the predicted quantile indexes that condition's training CDF.
    yhat = np.empty(len(df))
    missing = 0
    for k in np.unique(key):
        m = key == k
        ref = np.asarray(cp["cdfs"].get(k))
        if ref is None or ref.size == 0:
            yhat[m] = df.q.to_numpy()[m]
            missing += int(m.sum())
            continue
        pos_ = np.clip((df.q.to_numpy()[m] * (len(ref) - 1)).astype(int), 0, len(ref) - 1)
        yhat[m] = ref[pos_]
    if missing:
        logger.warning("%d rows had no stored CDF; left on the raw quantile scale", missing)

    logger.info("")
    logger.info("%-34s %8s %8s", "scope", "raw q", "inverted")
    for lab, sel in (("ALL", np.ones(len(df), bool)),
                     ("Liu (hsu2026)", (df.source_study == "hsu2026").to_numpy()),
                     ("Kim (deepprime)", (df.source_study == "deepprime").to_numpy())):
        y = df.true_efficiency.to_numpy()[sel]
        logger.info("%-34s %8.4f %8.4f", lab,
                    spearmanr(y, df.q.to_numpy()[sel]).statistic,
                    spearmanr(y, yhat[sel]).statistic)

    # Within-condition ranking is what the objective actually targets, so report it too.
    kim = df[df.source_study == "deepprime"].copy()
    kim["_yh"] = yhat[(df.source_study == "deepprime").to_numpy()]
    ws = [spearmanr(s.true_efficiency, s._yh).statistic
          for _, s in kim.groupby(["cell_type", "pe_type"]) if len(s) >= 100]
    logger.info("")
    logger.info("Kim mean WITHIN-condition Spearman: %.4f  (%d conditions)", np.nanmean(ws), len(ws))

    Path("results/round7").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"record_id": df.record_id, "true_efficiency": df.true_efficiency,
                  "source_study": df.source_study, "cell_type": df.cell_type,
                  "pe_type": df.pe_type, "predicted_efficiency": yhat,
                  "raw_quantile": df.q}).to_parquet(
        f"results/round7/ctxprimary_{args.dev_fold_col}.parquet", index=False)
    logger.info("wrote results/round7/ctxprimary_%s.parquet", args.dev_fold_col)


if __name__ == "__main__":
    main()
