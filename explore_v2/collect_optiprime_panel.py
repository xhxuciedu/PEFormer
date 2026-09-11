"""Join OptiPrime's chunked panel predictions into one record_id -> prediction table.

`PREDICT_PE.py` writes `predictions.csv` (one column per weight directory plus their mean,
which is how OptiPrime itself ensembles) and `joined_df.csv` (its own post-preprocessing
frame, which carries our `record_id` through). They are row-aligned, which is checked here
rather than assumed.

Usage: .venv/bin/python explore_v2/collect_optiprime_panel.py [--work DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_WORK = Path("/srv/disk01/xhx/tmp/claude-8385/-srv-disk01-xhx-git-PEFormer/"
                    "c07d2d81-0766-40c2-8caf-61371c8f16e6/scratchpad/op_panel")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--out", type=Path,
                    default=Path("explore_v2/cache/optiprime_panel_predictions.parquet"))
    args = ap.parse_args()

    frames = []
    for dd in sorted(args.work.glob("chunk*")):
        outs = sorted(dd.glob("predictions_*"))
        if not outs:
            print(f"{dd.name}: no prediction directory, skipped")
            continue
        o = outs[-1]
        pr = pd.read_csv(o / "predictions.csv")
        jd = pd.read_csv(o / "joined_df.csv")
        assert len(pr) == len(jd), f"{dd.name}: {len(pr)} predictions vs {len(jd)} rows"
        assert "record_id" in jd.columns, f"{dd.name}: joined_df lost record_id"
        n_models = len([c for c in pr.columns if c.startswith("model_")])
        frames.append(pd.DataFrame({"record_id": jd.record_id.to_numpy(),
                                    "op": pr.mean_pred.to_numpy()}))
        print(f"{dd.name}: {len(pr):,} rows, {n_models} weight directories averaged")
    if not frames:
        raise SystemExit("no chunks produced predictions")
    out = pd.concat(frames, ignore_index=True)
    dup = out.record_id.duplicated().sum()
    assert dup == 0, f"{dup} duplicated record_id across chunks"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(out):,} rows, "
          f"prediction range {out.op.min():.3g} to {out.op.max():.3g})")


if __name__ == "__main__":
    main()
