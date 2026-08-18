"""Precompute and cache the featurized corpus for fast training startup."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import featurize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("featurize_corpus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/optiprime_full_297962.parquet")
    ap.add_argument("--out", default="data/processed/featurized_corpus.npz")
    ap.add_argument("--vocab", default="data/processed/context_vocab.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.corpus)
    logger.info("loaded %d rows", len(df))

    vocab = ContextVocab.fit(df)
    logger.info("context vocab sizes: %s", vocab.sizes())
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    vocab.save(Path(args.vocab))

    corpus = featurize(df, vocab)
    out = Path(args.out)
    np.savez_compressed(
        out,
        edit_ids=corpus.edit_ids,
        peg_nuc_ids=corpus.peg_nuc_ids,
        peg_seg_ids=corpus.peg_seg_ids,
        target=corpus.target,
        target_indel=corpus.target_indel,
        group_key=corpus.group_key,
        fold=corpus.fold,
        record_id=corpus.record_id,
        **{f"ctx_{k}": v for k, v in corpus.context_ids.items()},
    )
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
