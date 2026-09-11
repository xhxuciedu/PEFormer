"""E14 - does fixing the ranking loss's grouping improve the decision it is meant to rank?

E13 found that the pairwise ranking term (loss coefficient 0.25) groups rows by the raw
window pair. E17a's replay of the real sampler shows what that costs: 100.0% of its sampled
pairs compare two designs of identical RTT length, and it consumes 15,695 sampled pair
instances per epoch against 62,028 under the canonical key. This evaluates the two arms E13 defines: the unchanged recipe, and the same
recipe with `group_key` replaced by the canonical decision group. Same architecture, same
seed, same code, one changed input array.

Evaluation surface: the locked fold 0, which neither arm trains on and neither uses for
early stopping (that is fold 1). Two endpoints:

* pooled Spearman, the manuscript's headline metric, as a "did anything break" check;
* the canonical fixed-allele decision, which is what the ranking term is supposed to serve
  and what E02 established as the estimand a user faces.

The reserved panel is deliberately **not** used here. It was spent on its one pre-declared
confirmatory comparison in E11, and comparing two freshly trained arms on it would make it
development data.

Fold 0 has been examined many times across nine rounds, so this comparison is development
evidence about a training choice, not a confirmatory result.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_v2/e14_eval_regroup.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

ARMS = {"control (current ranking key)": "e13_ctrl",
        "canonical decision-group key": "e13_canon"}


@torch.no_grad()
def score(ckpt: Path, corpus, rows: np.ndarray, device: str, bs: int) -> np.ndarray:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    ds = PEDataset(corpus)
    out = []
    for s in range(0, len(rows), bs):
        b = {k: v.to(device) for k, v in
             collate([ds[i] for i in rows[s:s + bs]]).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = m(b)
        out.append(m.efficiency_from_output(o).float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def decision_metrics(d: pd.DataFrame, col: str) -> dict:
    """Fixed-allele decision on fold 0: pairwise accuracy, achieved efficiency, regret."""
    rho, acc, sel, reg, best, rnd = [], [], [], [], [], []
    for _, s in d.groupby("decision_group", observed=True):
        if s.design_key.nunique() < 2:
            continue
        y = s.edited.to_numpy()
        if np.ptp(y) == 0:
            continue
        p = s[col].to_numpy()
        pick = int(np.argmax(p))
        acc.append(float(y[pick] == y.max()))
        sel.append(float(y[pick]))
        reg.append(float(y.max() - y[pick]))
        best.append(float(y.max()))
        rnd.append(float(y.mean()))
        if len(y) > 2:
            rho.append(C.spearman(p, y))
    return {"groups": len(acc), "accuracy": float(np.mean(acc)),
            "achieved_efficiency": float(np.mean(sel)), "regret": float(np.mean(reg)),
            "oracle": float(np.mean(best)), "random": float(np.mean(rnd)),
            "mean_spearman_ge3": float(np.mean(rho)) if rho else np.nan,
            "regret_share_of_random_removed": float(
                1 - np.mean(reg) / max(np.mean(best) - np.mean(rnd), 1e-12))}


def paired_bootstrap(d: pd.DataFrame, a: str, b: str, seed: int, n_boot: int = 2000) -> dict:
    """Paired difference in per-group achieved efficiency and accuracy, clustered on spacer."""
    rows = []
    for _, s in d.groupby("decision_group", observed=True):
        if s.design_key.nunique() < 2:
            continue
        y = s.edited.to_numpy()
        if np.ptp(y) == 0:
            continue
        r = {"spacer": s.spacer.iloc[0], "best": float(y.max())}
        for col in (a, b):
            pick = int(np.argmax(s[col].to_numpy()))
            r[f"{col}_sel"] = float(y[pick])
            r[f"{col}_acc"] = float(y[pick] == y.max())
        rows.append(r)
    t = pd.DataFrame(rows)
    out = {}
    for metric in ("sel", "acc"):
        diff = (t[f"{a}_{metric}"] - t[f"{b}_{metric}"]).to_numpy()
        uniq, inv = np.unique(t.spacer.to_numpy(), return_inverse=True)
        buckets = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        rng = np.random.default_rng(seed)
        vals = np.array([diff[np.concatenate([buckets[j] for j in
                                              rng.integers(0, len(uniq), len(uniq))])].mean()
                         for _ in range(n_boot)])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        frac = float((vals > 0).mean())
        out[metric] = {"observed": float(diff.mean()), "ci95": [float(lo), float(hi)],
                       "two_sided_p": float(max(2 * min(frac, 1 - frac), 1 / n_boot)),
                       "n_groups": int(len(t)), "n_spacer_clusters": int(len(uniq))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = load_featurized(str(ROOT / "data/processed/featurized_official.npz"), vocab)
    fold = np.asarray(corpus.fold)
    rows = np.where(fold == 0)[0]
    rid = np.asarray(corpus.record_id)[rows]

    man = pd.read_parquet(C.require_manifest())
    d = pd.DataFrame({"record_id": rid}).merge(man, on="record_id", validate="1:1")

    found = {}
    for label, stem in ARMS.items():
        cks = sorted((ROOT / "checkpoints").glob(f"{stem}_*/best.pt"))
        if not cks:
            print(f"{label}: no checkpoint yet ({stem}_*)", flush=True)
            continue
        ck = cks[-1]
        d[stem] = score(ck, corpus, rows, device, args.batch_size)
        found[label] = str(ck.relative_to(ROOT))
        print(f"{label}: scored from {ck.parent.name}", flush=True)
    if len(found) < 2:
        raise SystemExit("both arms must be trained before this comparison is meaningful")

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "checkpoints": found,
           "surface": "locked fold 0 (20,509 rows); neither arm trains on it or "
                      "early-stops on it",
           "pooled_spearman": {label: C.spearman(d[stem].to_numpy(), d.edited.to_numpy())
                               for label, stem in ARMS.items() if label in found},
           "decision": {label: decision_metrics(d, stem)
                        for label, stem in ARMS.items() if label in found},
           "paired_canon_minus_control": paired_bootstrap(
               d, "e13_canon", "e13_ctrl", args.seed)}
    d[["record_id", "decision_group", "design_key", "spacer", "edited",
       "e13_ctrl", "e13_canon"]].to_parquet(C.CACHE / "e14_fold0_scores.parquet", index=False)
    C.write_outputs("e14_eval_regroup", res, render(res))


def render(r: dict) -> str:
    L = ["# E14 - fixing the ranking loss's grouping\n",
         f"Surface: {r['surface']}. Same architecture, same seed, same code; the arms differ "
         "only in which rows the pairwise ranking loss may compare.\n",
         "## Pooled Spearman, the manuscript's headline metric\n",
         "| arm | pooled rho |\n|---|---:|"]
    for k, v in r["pooled_spearman"].items():
        L.append(f"| {k} | {v:.4f} |")
    L.append("\n## The fixed-allele decision the ranking term is meant to serve\n")
    first = next(iter(r["decision"].values()))
    L.append(f"{first['groups']:,} scorable decision groups on fold 0 (oracle "
             f"{first['oracle']:.4f}, random {first['random']:.4f}).\n")
    L.append("| arm | accuracy | achieved efficiency | regret | share of random regret removed |\n|---|---:|---:|---:|---:|")
    for k, v in r["decision"].items():
        L.append(f"| {k} | {v['accuracy']:.4f} | {v['achieved_efficiency']:.4f} | "
                 f"{v['regret']:.5f} | {v['regret_share_of_random_removed']:.1%} |")
    p = r["paired_canon_minus_control"]
    L.append("\n## Paired, clustered on protospacer\n")
    L.append("| difference (canonical key - control) | value | 95% CI | p |\n|---|---:|---|---:|")
    for k, lab in (("acc", "best-design accuracy"), ("sel", "achieved efficiency")):
        v = p[k]
        L.append(f"| {lab} | {v['observed']:+.5f} | [{v['ci95'][0]:+.5f}, "
                 f"{v['ci95'][1]:+.5f}] | {v['two_sided_p']:.3g} |")
    L.append(f"\n{p['acc']['n_groups']:,} groups over {p['acc']['n_spacer_clusters']:,} "
             "protospacer clusters.\n")
    L.append("Fold 0 has been examined many times across nine rounds, so this is development "
             "evidence about a training choice, not a confirmatory result. The reserved panel "
             "is deliberately not used: it was spent on its one pre-declared comparison in "
             "E11.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
