"""Phase 0: reconcile every quantity that appears more than once in the manuscript.

Traces each duplicated or artifact-backed number in
`reports/paper/pe_rankformer_paper.tex` back to the file that produces it, and reports
agreement or disagreement. Writes a machine-readable result and a markdown summary.

Usage: python revision/audit_00_consistency.py [--seed 20260903]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

TEX = Path("reports/paper/pe_rankformer_paper.tex")
CORPUS = Path("data/processed/optiprime_official_318471.parquet")
DEV_ASSIGN = Path("data/processed/round3_dev_assignments.parquet")
HELD = Path("results/round5/heldout_calibrated.parquet")
H2H = Path("results/heldout_full_head_to_head.parquet")
OUT_JSON = Path("revision/00_consistency_audit.json")
OUT_MD = Path("revision/00_consistency_audit.md")


def sha256(p: Path, limit: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            if f.tell() > limit:
                break
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()[:12]
    except Exception:
        return "unknown"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------- #
# suspect 1: zero-mass populations
# --------------------------------------------------------------------------- #
def audit_zero_mass(df: pd.DataFrame, dev: pd.DataFrame) -> dict:
    tr, ho = df[df.fold != 0], df[df.fold == 0]
    devrows = df[df.record_id.isin(dev.record_id)]
    # the matched development folds are the union of the three fold columns' val sets
    # The dev-fold columns are string-valued 'train'/'val'. The reported development
    # numbers are out-of-fold, i.e. computed on the 'val' rows of each fold, so that is
    # the population any "development-fold" percentage refers to.
    per_fold = {}
    for c in ("round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"):
        ids = dev.loc[dev[c] == "val", "record_id"]
        sub = df[df.record_id.isin(ids)]
        kim = sub[sub.source_study == "deepprime"]
        per_fold[c] = {"n": int(len(sub)), "zero_mass": float((sub.edited == 0).mean()),
                       "kim_n": int(len(kim)),
                       "kim_zero_mass": float((kim.edited == 0).mean())}
    per_fold["mean_of_three_val_folds"] = {
        "n": int(np.mean([v["n"] for k, v in per_fold.items()])),
        "zero_mass": float(np.mean([v["zero_mass"] for k, v in per_fold.items()])),
        "kim_n": int(np.mean([v["kim_n"] for k, v in per_fold.items()])),
        "kim_zero_mass": float(np.mean([v["kim_zero_mass"] for k, v in per_fold.items()])),
    }
    return {
        "training_297962": {"n": int(len(tr)), "zero_mass": float((tr.edited == 0).mean()),
                            "kim_zero_mass": float((tr[tr.source_study == "deepprime"].edited == 0).mean())},
        "heldout_20509": {"n": int(len(ho)), "zero_mass": float((ho.edited == 0).mean()),
                          "kim_zero_mass": float((ho[ho.source_study == "deepprime"].edited == 0).mean())},
        "dev_pool_all_297962": {"n": int(len(devrows)), "zero_mass": float((devrows.edited == 0).mean()),
                                "note": "the dev assignment file covers all training rows; "
                                        "the reported dev numbers are the val subsets below"},
        "per_dev_fold": per_fold,
    }


# --------------------------------------------------------------------------- #
# suspect 2: the three Kim numbers
# --------------------------------------------------------------------------- #
def audit_kim(df: pd.DataFrame) -> dict:
    held = pd.read_parquet(HELD)
    hk = held[held.source_study == "deepprime"]
    out = {
        "heldout_frozen_ensemble": {
            "n": int(len(hk)), "rho": spearman(hk.predicted_efficiency.to_numpy(), hk.true_efficiency.to_numpy()),
            "what": "5-member rank-average ensemble, official held-out rows"},
        "heldout_best_single_member": {
            "n": int(len(hk)), "rho": spearman(hk.member_ordSSM.to_numpy(), hk.true_efficiency.to_numpy()),
            "what": "ordinal+S4D member alone, official held-out rows"},
    }
    devs = []
    for f in range(3):
        p = Path(f"results/round3/dev_recalibration/predictions_r4p2_ordSSM_oof_round3_dev_fold_{f}.parquet")
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        k = d[d.source_study == "deepprime"]
        devs.append({"fold": f, "n": int(len(k)),
                     "rho": spearman(k.predicted_efficiency.to_numpy(), k.true_efficiency.to_numpy())})
    out["dev_ordSSM_per_fold"] = devs
    if devs:
        out["dev_ordSSM_mean_of_folds"] = {
            "n_mean": int(np.mean([d["n"] for d in devs])),
            "rho": float(np.mean([d["rho"] for d in devs])),
            "what": "ordinal+S4D out-of-fold, mean over the 3 matched development folds"}
    return out


# --------------------------------------------------------------------------- #
# suspect 4: ordinal threshold counts actually stored in checkpoints
# --------------------------------------------------------------------------- #
def audit_thresholds() -> dict:
    def th_of(mc):
        if mc is None:
            return None
        return mc.get("ordinal_thresholds") if isinstance(mc, dict) else getattr(mc, "ordinal_thresholds", None)

    counts: collections.Counter = collections.Counter()
    official: collections.Counter = collections.Counter()
    for p in sorted(Path("checkpoints").glob("*/best.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        th = th_of(ck.get("model_config"))
        if not th:
            continue
        counts[len(th)] += 1
        # the frozen reported system: the five official-fold checkpoints of each member
        if re.match(r"r4p2_(ordSSM|ordA|ordB|ordC)_cv[1-5]_", p.parent.name):
            official[len(th)] += 1
    return {"all_ordinal_checkpoints": dict(counts),
            "official_frozen_member_checkpoints": dict(official)}


# --------------------------------------------------------------------------- #
# duplicated-number sweep over the manuscript
# --------------------------------------------------------------------------- #
def sweep_duplicates(tex: str) -> list[dict]:
    body = "\n".join(l for l in tex.split("\n") if not l.strip().startswith("%"))
    # 4-decimal quantities are the paper's metric convention; also catch percentages
    nums = collections.defaultdict(list)
    for m in re.finditer(r"(?<![\d.])(\d\.\d{4})(?![\d])", body):
        line = body[:m.start()].count("\n") + 1
        nums[m.group(1)].append(line)
    for m in re.finditer(r"(\d{1,3}\.\d)\\%", body):
        line = body[:m.start()].count("\n") + 1
        nums[m.group(1) + "%"].append(line)
    return [{"value": v, "occurrences": len(ls), "tex_lines": ls}
            for v, ls in sorted(nums.items(), key=lambda kv: -len(kv[1])) if len(ls) > 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    np.random.seed(args.seed)

    df = pd.read_parquet(CORPUS, columns=["record_id", "edited", "fold", "source_study"])
    dev = pd.read_parquet(DEV_ASSIGN)

    result = {
        "git_commit": git_commit(),
        "seed": args.seed,
        "inputs": {str(p): sha256(p) for p in (TEX, CORPUS, DEV_ASSIGN, HELD, H2H) if p.exists()},
        "suspect_1_zero_mass": audit_zero_mass(df, dev),
        "suspect_2_kim_numbers": audit_kim(df),
        "suspect_3_kim_zero_fraction": "covered by suspect_1 (per-population Kim zero-mass)",
        "suspect_4_ordinal_thresholds": audit_thresholds(),
        "duplicated_numbers_in_manuscript": sweep_duplicates(TEX.read_text()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    z = result["suspect_1_zero_mass"]
    k = result["suspect_2_kim_numbers"]
    t = result["suspect_4_ordinal_thresholds"]
    print("=== suspect 1: zero-mass ===")
    for name, v in z.items():
        if name == "per_dev_fold":
            for f, vv in v.items():
                print(f"  {f:24s} n={vv['n']:6d} zero={vv['zero_mass']:.4f} "
                      f"kim_n={vv['kim_n']:6d} kim_zero={vv['kim_zero_mass']:.4f}")
        else:
            print(f"  {name:24s} n={v['n']:6d} zero={v['zero_mass']:.4f}"
                  + (f" kim_zero={v['kim_zero_mass']:.4f}" if "kim_zero_mass" in v else ""))
    print("\n=== suspect 2: Kim ===")
    for name, v in k.items():
        if isinstance(v, list):
            for vv in v:
                print(f"  dev fold {vv['fold']}: n={vv['n']} rho={vv['rho']:.4f}")
        else:
            print(f"  {name:28s} n={v.get('n', v.get('n_mean'))} rho={v['rho']:.4f}  <- {v['what']}")
    print("\n=== suspect 4: thresholds ===")
    print(f"  all ordinal checkpoints : {t['all_ordinal_checkpoints']}")
    print(f"  official frozen members : {t['official_frozen_member_checkpoints']}")
    print(f"\nduplicated 4-dp / percentage quantities: {len(result['duplicated_numbers_in_manuscript'])}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
