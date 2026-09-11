"""Out-of-fold frozen design embeddings from the ordinal-S4D backbone.

The research plan's priority 1 starts from `eta(d,c) = f_theta(d) + a_c + u_theta(d)^T z_c`
with `f_theta` frozen. This script produces the `f_theta(d)` side of that: the pooled
sequence representation taken **before** FiLM, which is a pure function of the edit and
the pegRNA and carries no context information at all.

Out of fold, by construction. Round 4 phase 2 trained five ordinal-S4D checkpoints, one
per official fold (`--val-fold k`, so checkpoint k trained on folds {1..5}\\{k} and never
on fold k or on the held-out fold 0). Every row here is embedded and scored by the one
checkpoint that held its own fold out, so no row's label influenced the weights that
represent it. Fold 0 is embedded with the cv1 checkpoint; all five are equally clean for
fold 0, and using one keeps the fold-0 representation internally comparable.

Outputs to explore_v2/cache/:
  embeddings.npy       (318471, 768) float16, pre-FiLM pooled representation
  embed_index.parquet  record_id, fold, checkpoint, oof_pred (efficiency), row order

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 .venv/bin/python explore_v2/extract_embeddings.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

# The five checkpoints `run_round4_phase3.sh` selects for the OOF ordSSM member, with
# their recorded `val_fold` verified to cover folds 1-5 exactly once. `best.pt` is the
# best epoch *on the held-out fold*, so epoch choice saw one scalar per epoch from the
# fold being embedded; that is the residual and unavoidable contamination here.
CKPT = {1: "r4p2_ordSSM_cv1_1787193463", 2: "r4p2_ordSSM_cv2_1787193629",
        3: "r4p2_ordSSM_cv3_1787193470", 4: "r4p2_ordSSM_cv4_1787204733",
        5: "r4p2_ordSSM_cv5_1787204854"}


@torch.no_grad()
def run_one(ckpt_path: Path, corpus, rows: np.ndarray, device: str,
            batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    assert model.film is not None, "expected a late-FiLM model; pooled hook would be wrong"

    grab: list[torch.Tensor] = []

    def hook(_mod, inputs, _out):
        grab.append(inputs[0].detach().float().cpu())

    h = model.film.register_forward_hook(hook)
    ds = PEDataset(corpus)
    embs, preds = [], []
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        batch = {k: v.to(device) for k, v in collate([ds[i] for i in idx]).items()}
        grab.clear()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        preds.append(model.efficiency_from_output(out).float().cpu().numpy())
        embs.append(grab[0].numpy().astype(np.float16))
    h.remove()
    del model
    torch.cuda.empty_cache()
    return np.concatenate(embs), np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "explore_v2/cache")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = load_featurized(str(ROOT / "data/processed/featurized_official.npz"), vocab)
    fold = np.asarray(corpus.fold)
    rid = np.asarray(corpus.record_id)
    n = len(fold)

    E = np.zeros((n, 768), dtype=np.float16)
    P = np.zeros(n, dtype=np.float32)
    which = np.empty(n, dtype=object)
    for k, name in CKPT.items():
        rows = np.where(fold == k)[0] if k != 1 else np.where((fold == k) | (fold == 0))[0]
        e, p = run_one(ROOT / "checkpoints" / name / "best.pt", corpus, rows,
                       device, args.batch_size)
        assert e.shape[1] == 768, e.shape
        E[rows] = e
        P[rows] = p
        which[rows] = name
        print(f"fold {k}: {len(rows):,} rows via {name}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", E)
    pd.DataFrame({"record_id": rid, "fold": fold, "checkpoint": which,
                  "oof_pred": P, "row": np.arange(n)}).to_parquet(
        args.out_dir / "embed_index.parquet", index=False)
    print("saved", args.out_dir / "embeddings.npy", E.shape)


if __name__ == "__main__":
    main()
