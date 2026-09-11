"""E11 - the pre-declared comparison on the reserved panel.

This is the confirmatory run. `PREREGISTRATION_reserved_panel.md` declared the endpoints,
the eligibility rule, the stratification, the dependence unit and the primary comparison
before the panel was ever scored; E10 built and validated the inputs; nothing has been
tuned against it. Everything below is computed once.

Primary comparison: PE-RankFormer's ordinal-S4D backbone against OptiPrime, paired per
decision group, resampling target sites.

Exposure, which the review's section 2.7 correctly insisted on:

| model | trained on Kim LibSmall (the corpus) | trained on Kim's large library (this panel) |
|---|---|---|
| PE-RankFormer ordinal-S4D | yes | **no** |
| OptiPrime | yes | **no** |
| DeepPrime (released) | - | **yes** (`external/deepprime/train_base.py:26`) |

So DeepPrime is excluded as a comparator here; it is not a held-out baseline on its own
training library. The two models in the primary comparison are both unexposed.

Usage:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_v2/e11_score_reserved_panel.py
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
from extract_embeddings import CKPT  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, featurize  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

THRESHOLDS = (0.02, 0.05)   # "worth taking to the bench" on this library's scale
KS = (1, 3, 5)
EFFORT_KS = tuple(range(1, 9))   # for the screening-effort curve, on a fixed population


@torch.no_grad()
def score_ours(panel: pd.DataFrame, device: str, batch_size: int) -> pd.DataFrame:
    """Predicted efficiency from each of the five ordinal-S4D checkpoints, and their mean.

    The paper's `ordSSM` member is this five-checkpoint set. All five are unexposed to this
    panel, so their mean is the member as defined, and the individual columns are kept so
    the comparison cannot be accused of resting on a checkpoint choice.
    """
    df = panel.copy()
    df["edited"] = df.edited_frac
    df["indel"] = 0.0
    df["fold"] = 0
    df["target_name"] = df.protospacer
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    idx = np.arange(len(df))
    out = {}
    for k, name in CKPT.items():
        ck = torch.load(ROOT / "checkpoints" / name / "best.pt", map_location=device,
                        weights_only=False)
        model = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device).eval()
        model.load_state_dict(ck["model_state_dict"])
        preds = []
        for s in range(0, len(idx), batch_size):
            b = {kk: v.to(device) for kk, v in
                 collate([ds[i] for i in idx[s:s + batch_size]]).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o = model(b)
            preds.append(model.efficiency_from_output(o).float().cpu().numpy())
        out[f"ours_cv{k}"] = np.concatenate(preds)
        del model
        torch.cuda.empty_cache()
        print(f"  scored with {name}", flush=True)
    r = pd.DataFrame(out)
    r["ours"] = r.mean(axis=1)
    r["record_id"] = df.record_id.to_numpy()
    return r


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
def group_endpoints(y: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(-score, kind="stable")
    best = float(y.max())
    out = {}
    for k in KS:
        if len(y) < k:
            out[f"best_of_{k}"] = np.nan
            continue
        out[f"best_of_{k}"] = float(y[order[:k]].max())
    out["hit_rate_1"] = float(y[order[0]] == best)
    out["regret_1"] = best - float(y[order[0]])
    for t in THRESHOLDS:
        out[f"threshold_{t}_at_1"] = float(y[order[0]] >= t)
        out[f"threshold_{t}_at_3"] = float(y[order[:min(3, len(y))]].max() >= t) if len(y) >= 1 else np.nan
    return out


def screening_effort(panel: pd.DataFrame, models: list[str], min_depth: int = 8) -> dict:
    """How many candidates must be tested to reach what one nominated candidate delivers?

    The plan asks for "the budget needed to reach a fixed retrospective utility level". On a
    population of groups all at least `min_depth` deep -- so every budget is evaluated on
    exactly the same groups -- compute the mean achieved efficiency of the best of k
    nominated candidates for k = 1..8, for each ranker and for random selection. The
    practically legible statistic is then: how large must k be under random selection to
    match what the model achieves at k = 1?
    """
    from math import comb
    deep = panel.groupby("edit_key", observed=True).design_key.transform("nunique") >= min_depth
    p = panel[deep]
    curves: dict[str, list[float]] = {m: [] for m in models + ["rand"]}
    groups = list(p.groupby("edit_key", observed=True, sort=False))
    for k in EFFORT_KS:
        for m in models:
            vals = []
            for _, s in groups:
                y = s.edited_frac.to_numpy()
                o = np.argsort(-s[m].to_numpy(), kind="stable")[:k]
                vals.append(float(y[o].max()))
            curves[m].append(float(np.mean(vals)))
        vals = []
        for _, s in groups:
            ys = np.sort(s.edited_frac.to_numpy())
            n = len(ys)
            tot, prev, acc = comb(n, k), 0.0, 0.0
            for i in range(k, n + 1):
                cdf = comb(i, k) / tot
                acc += ys[i - 1] * (cdf - prev)
                prev = cdf
            vals.append(acc)
        curves["rand"].append(float(np.mean(vals)))
    out = {"min_depth": min_depth, "groups": int(len(groups)),
           "k": list(EFFORT_KS), "curves": curves}
    for m in models:
        target = curves[m][0]
        need = next((k for k, v in zip(EFFORT_KS, curves["rand"]) if v >= target), None)
        out[f"random_candidates_to_match_{m}_top1"] = need
        if "op" in models and m == "ours":
            t2 = curves["ours"][0]
            need_op = next((k for k, v in zip(EFFORT_KS, curves["op"]) if v >= t2), None)
            out["optiprime_candidates_to_match_ours_top1"] = need_op
    return out


def build_table(panel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for gid, s in panel.groupby("edit_key", observed=True, sort=False):
        y = s.edited_frac.to_numpy()
        n = len(y)
        rec = {"edit_key": gid, "site": s.protospacer.iloc[0], "n": n,
               "depth": int(s.design_key.nunique()),
               "edit_class": s.edit_type.iloc[0].rstrip("0123456789"),
               "oracle": float(y.max()), "informative": bool(y.max() > y.min()),
               "all_zero": bool(y.max() == 0), "mean_y": float(y.mean())}
        for m in models:
            for kk, vv in group_endpoints(y, s[m].to_numpy()).items():
                rec[f"{m}_{kk}"] = vv
        # exact expectation of a random pick, and of the best of k random picks
        ys = np.sort(y)
        from math import comb
        for k in KS:
            if n < k:
                rec[f"rand_best_of_{k}"] = np.nan
                continue
            tot, prev, acc = comb(n, k), 0.0, 0.0
            for i in range(k, n + 1):
                cdf = comb(i, k) / tot
                acc += ys[i - 1] * (cdf - prev)
                prev = cdf
            rec[f"rand_best_of_{k}"] = float(acc)
        rec["rand_hit_rate_1"] = 1.0 / n
        rec["rand_regret_1"] = float(y.max() - y.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def site_bootstrap(t: pd.DataFrame, col_a: str, col_b: str, seed: int,
                   n_boot: int = 2000) -> dict:
    """Paired difference, resampling target sites as the dependence unit."""
    d = (t[col_a] - t[col_b]).to_numpy()
    ok = np.isfinite(d)
    d, sites = d[ok], t.site.to_numpy()[ok]
    if d.size == 0:
        return {}
    uniq, inv = np.unique(sites, return_inverse=True)
    buckets = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([buckets[j] for j in rng.integers(0, len(uniq), len(uniq))])
        vals[i] = d[idx].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    frac = float((vals > 0).mean())
    return {"observed": float(d.mean()), "ci95": [float(lo), float(hi)],
            "two_sided_p": float(max(2 * min(frac, 1 - frac), 1.0 / n_boot)),
            "n_groups": int(d.size), "n_sites": int(len(uniq))}


def summarise(t: pd.DataFrame, models: list[str], seed: int, label: str) -> dict:
    out = {"stratum": label, "groups": int(len(t)), "sites": int(t.site.nunique()),
           "all_zero_groups": int(t.all_zero.sum()),
           "mean_oracle": float(t.oracle.mean()), "arms": {}}
    # The prereg requires that budget curves either hold the group population fixed or
    # report how it changes. best_of_k is undefined below depth k, so the eligible count is
    # reported for every k and the fixed-population view is the depth strata below.
    out["groups_with_depth_at_least"] = {str(k): int((t.depth >= k).sum()) for k in KS}
    for m in models + ["rand"]:
        a = {}
        for k in KS:
            c = f"{m}_best_of_{k}"
            if c in t:
                a[f"achieved_efficiency_at_{k}"] = float(t[c].mean())
        a["best_design_hit_rate"] = float(t[f"{m}_hit_rate_1"].mean())
        a["regret_at_1"] = float(t[f"{m}_regret_1"].mean())
        if m != "rand":
            for th in THRESHOLDS:
                a[f"threshold_{th}_success_at_1"] = float(t[f"{m}_threshold_{th}_at_1"].mean())
                a[f"threshold_{th}_success_at_3"] = float(t[f"{m}_threshold_{th}_at_3"].mean())
        out["arms"][m] = a
    if "ours" in models and "op" in models:
        out["paired_ours_minus_op"] = {
            "achieved_efficiency_at_1": site_bootstrap(t, "ours_best_of_1", "op_best_of_1", seed),
            "achieved_efficiency_at_3": site_bootstrap(t, "ours_best_of_3", "op_best_of_3", seed),
            "best_design_hit_rate": site_bootstrap(t, "ours_hit_rate_1", "op_hit_rate_1", seed),
            "regret_at_1": site_bootstrap(t, "ours_regret_1", "op_regret_1", seed),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--op-predictions", type=Path, default=None,
                    help="CSV/parquet of OptiPrime predictions with record_id + prediction")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    cache = C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet"
    if cache.exists():
        ours = pd.read_parquet(cache)
        print(f"reusing cached predictions from {cache.name}", flush=True)
    else:
        ours = score_ours(panel, device, args.batch_size)
        ours.to_parquet(cache, index=False)
    panel = panel.merge(ours, on="record_id", validate="1:1")
    models = ["ours"]

    op_note = "not available"
    if args.op_predictions and args.op_predictions.exists():
        op = (pd.read_parquet(args.op_predictions)
              if args.op_predictions.suffix == ".parquet"
              else pd.read_csv(args.op_predictions))
        col = [c for c in op.columns if c != "record_id"][0]
        panel = panel.merge(op[["record_id", col]].rename(columns={col: "op"}),
                            on="record_id", how="left", validate="1:1")
        cov = float(panel.op.notna().mean())
        op_note = f"joined, coverage {cov:.4f}"
        # a tool that cannot score a candidate must be visible, not silently dropped
        panel = panel[panel.op.notna()]
        models.append("op")

    t = build_table(panel, models)
    t.to_csv(C.OUT / "e11_panel_group_table.csv", index=False)

    # Sanity check before any endpoint: a pooled rank correlation in a plausible range is
    # the cheapest evidence that the sequence reconstruction and the assigned context are
    # sane. A reconstruction error would show up here as a near-zero correlation.
    pooled = {m: C.spearman(panel[m].to_numpy(), panel.edited_frac.to_numpy())
              for m in models}
    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "panel_rows_scored": int(len(panel)),
           "pooled_spearman_sanity_check": pooled,
           "optiprime": op_note,
           "exposure": {"pe_rankformer_ordinal_s4d": "not trained on this panel",
                        "optiprime": "not trained on this panel",
                        "deepprime_released": "TRAINED on this panel's source file; "
                                              "excluded as a comparator"},
           "strata": []}
    elig = t
    res["strata"].append(summarise(elig, models, args.seed, "all eligible groups (depth >=2)"))
    res["strata"].append(summarise(elig[elig.informative], models, args.seed,
                                   "informative groups (non-constant outcome)"))
    for lo, hi, lab in [(2, 2, "depth 2"), (3, 4, "depth 3-4"), (5, 7, "depth 5-7"),
                        (8, 99, "depth 8+")]:
        s = elig[(elig.depth >= lo) & (elig.depth <= hi)]
        if len(s) >= 100:
            res["strata"].append(summarise(s, models, args.seed, lab))
    # How much the choice is worth. A ranking advantage on groups whose candidates are
    # indistinguishable is not worth anything; the question is whether the advantage is
    # there when the spread is large.
    # decision value = oracle minus the expectation of a random pick, i.e. exactly what a
    # perfect chooser would gain over no information at all in that group
    elig = elig.assign(decision_value=elig.oracle - elig.mean_y)
    for lo, hi, lab in [(0.0, 0.005, "decision worth < 0.005"),
                        (0.005, 0.02, "decision worth 0.005-0.02"),
                        (0.02, 0.05, "decision worth 0.02-0.05"),
                        (0.05, 1.01, "decision worth >= 0.05")]:
        s_ = elig[(elig.decision_value >= lo) & (elig.decision_value < hi)]
        if len(s_) >= 100:
            res["strata"].append(summarise(s_, models, args.seed, lab))
    for cls in ("sub", "ins", "del"):
        s = elig[elig.edit_class == cls]
        if len(s) >= 100:
            res["strata"].append(summarise(s, models, args.seed, f"edit type {cls}"))
    res["screening_effort"] = screening_effort(panel, models)
    C.write_outputs("e11_score_reserved_panel", res, render(res))


def render(r: dict) -> str:
    L = ["# E11 - the pre-declared comparison on the reserved panel\n",
         f"{r['panel_rows_scored']:,} candidate measurements scored. Canonicaliser version "
         f"{r['canon_version']}. OptiPrime: {r['optiprime']}.\n",
         "Pooled Spearman on the panel, as a reconstruction sanity check (not an endpoint): "
         + ", ".join(f"{k} {v:.4f}" for k, v in
                     r["pooled_spearman_sanity_check"].items()) + ".\n",
         "Exposure: PE-RankFormer's ordinal-S4D backbone and OptiPrime were both trained on "
         "Kim's LibSmall files and **neither** on this panel's source library. Released "
         "DeepPrime was trained on it and is therefore excluded as a comparator here.\n"]
    L.append("Reading the table: a group whose designs tie at the maximum gives every "
             "predictor a hit, which is why random selection's best-design hit rate is well "
             "above 1/depth -- 5,807 groups have all designs at exactly zero. The "
             "pre-registration requires tied optima to receive full credit and an all-zero "
             "tie not to be presented as success, so the all-eligible stratum is the "
             "deployment number and the informative stratum is the discrimination number, "
             "and both are reported.\n")
    prim = r["strata"][0]
    for st in r["strata"]:
        L.append(f"\n## {st['stratum']}\n")
        L.append(f"{st['groups']:,} groups over {st['sites']:,} target sites; "
                 f"{st['all_zero_groups']:,} have every design at exactly zero; oracle "
                 f"achieves {st['mean_oracle']:.4f}. Groups eligible at each budget: "
                 f"{st['groups_with_depth_at_least']}.\n")
        cols = [f"achieved_efficiency_at_{k}" for k in KS]
        L.append("| predictor | " + " | ".join(f"achieved @{k}" for k in KS) +
                 " | best-design hit rate | regret @1 |\n" + "|---" * (len(cols) + 3) + "|")
        for m, lab in (("rand", "random choice"), ("op", "OptiPrime"),
                       ("ours", "PE-RankFormer ordinal-S4D")):
            if m not in st["arms"]:
                continue
            a = st["arms"][m]
            cells = [f"{a[c]:.4f}" if c in a and np.isfinite(a[c]) else "-" for c in cols]
            L.append(f"| {lab} | " + " | ".join(cells) +
                     f" | {a['best_design_hit_rate']:.4f} | {a['regret_at_1']:.5f} |")
        if "paired_ours_minus_op" in st:
            L.append("\n| paired difference (ours - OptiPrime) | value | 95% CI | p |\n|---|---:|---|---:|")
            for k, v in st["paired_ours_minus_op"].items():
                if not v:
                    continue
                L.append(f"| {k.replace('_', ' ')} | {v['observed']:+.5f} | "
                         f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | {v['two_sided_p']:.3g} |")
    se = r.get("screening_effort")
    if se:
        L.append(f"\n## Screening effort, fixed population (depth >= {se['min_depth']})\n")
        L.append(f"{se['groups']:,} decision groups, every budget scored on the same groups. "
                 "Mean achieved efficiency of the best of k nominated candidates.\n")
        L.append("| k | " + " | ".join({"rand": "random", "op": "OptiPrime",
                                        "ours": "PE-RankFormer"}.get(m, m)
                                       for m in se["curves"]) + " |")
        L.append("|---" * (len(se["curves"]) + 1) + "|")
        for i, k in enumerate(se["k"]):
            L.append(f"| {k} | " + " | ".join(f"{v[i]:.4f}" for v in se["curves"].values()) + " |")
        n = se.get("random_candidates_to_match_ours_top1")
        L.append(f"\n**Random selection needs {n if n else '>8'} candidates to match what "
                 "PE-RankFormer's single nominated candidate achieves.**")
        if se.get("optiprime_candidates_to_match_ours_top1"):
            L.append(f" OptiPrime needs "
                     f"{se['optiprime_candidates_to_match_ours_top1']}.")
        L.append("")
    if "paired_ours_minus_op" in prim:
        L.append("\nDependence unit is the target site, resampled with replacement; "
                 f"{prim['paired_ours_minus_op']['achieved_efficiency_at_1']['n_sites']:,} "
                 "sites in the primary stratum.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
