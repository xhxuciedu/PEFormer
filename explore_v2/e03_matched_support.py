"""E03 - is there a design-by-context interaction on matched candidate sets?

Research plan section 8A days 3-6 and section 2: run the interaction diagnostic on
*common support*, with a replicate-derived noise model and an additive counterexample,
before any model is asked to learn an interaction.

The quantity of interest is the quartet contrast

    D = [mu(d1,c1) - mu(d2,c1)] - [mu(d1,c2) - mu(d2,c2)]

for two alternative pegRNA designs d1, d2 implementing the **same canonical allele**, both
measured in both contexts c1, c2. A nonzero D is an interaction on the chosen scale. It is
not the same thing as a rank reversal, so both are reported.

Scale. Efficiency is a bounded proportion with a large exact-zero block, so differences on
the raw scale are heteroscedastic. The arcsine-root transform z = asin(sqrt(y)) is the
variance-stabilising choice for binomial-like data and is defined at zero; the 654
metadata-identical replicate groups confirm it flattens the mean-variance relationship
(correlation of group SD with group mean falls from 0.53 to 0.28). Raw-scale results are
reported alongside.

Nothing is trained here. The additive fit is a two-way mean decomposition used as a null,
not a predictor.

Usage: .venv/bin/python explore_v2/e03_matched_support.py [--max-pairs-per-cell 6]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402


def arcsine(y: np.ndarray) -> np.ndarray:
    return np.arcsin(np.sqrt(np.clip(y, 0.0, 1.0)))


def noise_model(d: pd.DataFrame) -> dict:
    """Pooled per-measurement SD on both scales, from metadata-identical replicates.

    Pooled by degrees of freedom rather than averaged over groups, so the 68 five-member
    groups carry their proper weight. 649 of the 654 groups are Kim rows, so this is a
    Kim-calibrated noise level applied elsewhere for want of anything better; that
    limitation is carried into every derived number below.
    """
    g = d.groupby("replicate_key", observed=True).agg(
        n=("edited", "size"), var_raw=("edited", "var"), var_z=("z", "var"),
        m=("edited", "mean"), study=("cond3", "first"))
    r = g[g.n > 1]
    dfree = (r.n - 1).sum()
    return {"n_groups": int(len(r)), "n_rows": int(r.n.sum()), "df": int(dfree),
            "sigma_raw": float(np.sqrt((r.var_raw * (r.n - 1)).sum() / dfree)),
            "sigma_z": float(np.sqrt((r.var_z * (r.n - 1)).sum() / dfree)),
            "median_group_sd_z": float(np.sqrt(r.var_z).median()),
            "frac_groups_from_kim": float((r.study.str.startswith("deepprime")).mean())}


def build_quartets(d: pd.DataFrame, max_pairs: int, rng: np.random.Generator) -> pd.DataFrame:
    """All (allele, design pair, context pair) quartets on exactly matched support.

    A quartet only exists if both designs were measured in both contexts, which is what
    makes the contrast a design comparison rather than a comparison of two differently
    populated candidate lists. Design pairs are capped per allele to keep alleles with
    many designs from dominating; the cap is applied with a fixed seed and the number of
    dropped pairs is reported.
    """
    recs = []
    n_capped = 0
    for allele, s in d.groupby("edit_key", observed=True):
        piv = s.pivot_table(index="design_key", columns="context_key", values="z",
                            aggfunc="mean")
        pivy = s.pivot_table(index="design_key", columns="context_key", values="edited",
                             aggfunc="mean")
        if piv.shape[0] < 2 or piv.shape[1] < 2:
            continue
        designs = list(piv.index)
        pairs = list(itertools.combinations(range(len(designs)), 2))
        if len(pairs) > max_pairs:
            n_capped += len(pairs) - max_pairs
            idx = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in sorted(idx)]
        Z = piv.to_numpy()
        Y = pivy.to_numpy()
        ctx = list(piv.columns)
        spacer = s.spacer.iloc[0]
        target = s.target_name.iloc[0]
        etype = s.edit_type.iloc[0]
        for i, j in pairs:
            # contexts where BOTH designs were measured
            ok = np.where(np.isfinite(Z[i]) & np.isfinite(Z[j]))[0]
            if ok.size < 2:
                continue
            for a, b in itertools.combinations(ok, 2):
                recs.append((allele, spacer, target, etype, designs[i], designs[j],
                             ctx[a], ctx[b],
                             Z[i, a] - Z[j, a], Z[i, b] - Z[j, b],
                             Y[i, a] - Y[j, a], Y[i, b] - Y[j, b],
                             Y[i, a], Y[j, a], Y[i, b], Y[j, b]))
    q = pd.DataFrame(recs, columns=["edit_key", "spacer", "target_name", "edit_type",
                                    "d1", "d2", "c1", "c2", "dz1", "dz2", "dy1", "dy2",
                                    "y_d1c1", "y_d2c1", "y_d1c2", "y_d2c2"])
    q["D_z"] = q.dz1 - q.dz2
    q["D_y"] = q.dy1 - q.dy2
    q.attrs["n_design_pairs_dropped_by_cap"] = n_capped
    return q


def variance_components(q: pd.DataFrame, sigma_z: float) -> dict:
    """Split the observed spread of the design contrast into main effect, interaction, noise.

    For one (allele, design pair) the contrast Delta(c) varies across contexts only through
    interaction and measurement error, so the mean within-pair variance across contexts
    estimates Var_interaction + 2*sigma^2. What is left of the total spread of Delta is the
    design-pair main effect.
    """
    long = pd.concat([
        q[["edit_key", "d1", "d2", "c1", "dz1"]].rename(columns={"c1": "c", "dz1": "delta"}),
        q[["edit_key", "d1", "d2", "c2", "dz2"]].rename(columns={"c2": "c", "dz2": "delta"}),
    ]).drop_duplicates(subset=["edit_key", "d1", "d2", "c"])
    within = long.groupby(["edit_key", "d1", "d2"], observed=True).delta.agg(["var", "size"])
    within = within[within["size"] >= 2]
    v_within = float((within["var"] * (within["size"] - 1)).sum() / (within["size"] - 1).sum())
    v_total = float(long.delta.var())
    v_noise_delta = 2 * sigma_z ** 2
    return {
        "n_design_pairs_with_2plus_contexts": int(len(within)),
        "var_delta_total": v_total,
        "var_delta_within_pair_across_contexts": v_within,
        "var_noise_in_delta": v_noise_delta,
        "var_interaction_estimate": float(v_within - v_noise_delta),
        "var_design_main_effect_estimate": float(v_total - v_within),
        "sd_interaction_estimate": float(np.sqrt(max(v_within - v_noise_delta, 0.0))),
        "sd_design_main_effect_estimate": float(np.sqrt(max(v_total - v_within, 0.0))),
        "interaction_share_of_delta_variance": float(max(v_within - v_noise_delta, 0.0) / v_total),
        "noise_share_of_delta_variance": float(v_noise_delta / v_total),
    }


def reversal_stats(q: pd.DataFrame, sigma_z: float, k: float = 2.0) -> dict:
    """Order-change rates, overall and restricted to contrasts noise can barely explain.

    A quartet's two contrasts each carry SD sqrt(2)*sigma. Calling an order change real
    requires both contrasts to exceed k standard errors; without that filter the reversal
    rate is dominated by pairs of designs that are simply indistinguishable.
    """
    se = np.sqrt(2.0) * sigma_z
    sign1, sign2 = np.sign(q.dz1), np.sign(q.dz2)
    rev = (sign1 * sign2) < 0
    both_big = (q.dz1.abs() > k * se) & (q.dz2.abs() > k * se)
    out = {"n_quartets": int(len(q)),
           "reversal_rate_all": float(rev.mean()),
           "n_both_contrasts_significant": int(both_big.sum()),
           "reversal_rate_supported": float(rev[both_big].mean()) if both_big.any() else np.nan,
           "se_of_one_contrast": float(se), "k": k}
    # practical size: a reversal only matters if the better design changes by enough to
    # care about on the raw efficiency scale
    for thr in (0.02, 0.05, 0.10):
        mat = both_big & (q.dy1.abs() > thr) & (q.dy2.abs() > thr)
        out[f"n_supported_and_gap_gt_{thr}"] = int(mat.sum())
        out[f"reversal_rate_supported_and_gap_gt_{thr}"] = (
            float(rev[mat].mean()) if mat.any() else np.nan)
    return out


def additive_null(q: pd.DataFrame, sigma_z: float, seed: int, n_sim: int) -> dict:
    """Parametric counterexample: what these statistics look like with no interaction.

    Under *any* additive latent model the quartet contrast D is exactly measurement error,
    so SD(D) = 2*sigma analytically and no simulation is needed for that. Reversal rates
    are different: they depend on how large the true design contrasts are, so the null must
    reproduce the observed contrast sizes or it will manufacture reversals that the data
    would never show. The null therefore conditions on each quartet's own additive
    estimate of its contrast, Delta_hat = (Delta_1 + Delta_2)/2, and re-observes it twice:

        Delta_k* = Delta_hat + sqrt(2)*sigma*eps_k,  k = 1, 2

    That leaves the null over-dispersed by sigma^2 relative to the truth while carrying no
    interaction at all, which makes the comparison conservative as long as the interaction
    variance is well above sigma^2 (it is, by an order of magnitude).
    """
    rng = np.random.default_rng(seed)
    delta_hat = ((q.dz1 + q.dz2) / 2).to_numpy()
    se = np.sqrt(2.0) * sigma_z
    keys = ["reversal_rate_all", "reversal_rate_supported", "mean_abs_D_z", "sd_D_z"]
    acc = {k: [] for k in keys}
    for _ in range(n_sim):
        e = rng.normal(0.0, se, size=(len(q), 2))
        dz1, dz2 = delta_hat + e[:, 0], delta_hat + e[:, 1]
        D = dz1 - dz2
        rev = np.sign(dz1) * np.sign(dz2) < 0
        big = (np.abs(dz1) > 2 * se) & (np.abs(dz2) > 2 * se)
        acc["reversal_rate_all"].append(float(rev.mean()))
        acc["reversal_rate_supported"].append(float(rev[big].mean()) if big.any() else np.nan)
        acc["mean_abs_D_z"].append(float(np.abs(D).mean()))
        acc["sd_D_z"].append(float(D.std()))
    out = {"n_sim": n_sim, "analytic_sd_D_z": float(2 * sigma_z),
           "analytic_mean_abs_D_z": float(2 * sigma_z * np.sqrt(2 / np.pi))}
    for k, v in acc.items():
        arr = np.asarray(v, dtype=float)
        out[k] = {"mean": float(np.nanmean(arr)),
                  "p2.5": float(np.nanpercentile(arr, 2.5)),
                  "p97.5": float(np.nanpercentile(arr, 97.5))}
    return out


def noise_sensitivity(q: pd.DataFrame, sigma_z: float) -> list[dict]:
    """How wrong would the replicate-based noise estimate have to be?

    The interaction estimate is a difference between an observed within-pair variance and
    2*sigma^2. The replicate groups are 99% Kim and may not represent every assay, so the
    honest report is the multiple of sigma at which the estimate reaches zero.
    """
    rows = []
    for mult in (1.0, 1.5, 2.0, 2.5, 3.0):
        vc = variance_components(q, sigma_z * mult)
        rows.append({"sigma_multiplier": mult, "sigma_z": sigma_z * mult,
                     "var_interaction_estimate": vc["var_interaction_estimate"],
                     "sd_interaction_estimate": vc["sd_interaction_estimate"],
                     "interaction_share": vc["interaction_share_of_delta_variance"]})
    vc1 = variance_components(q, sigma_z)
    v_within = vc1["var_delta_within_pair_across_contexts"]
    rows.append({"sigma_multiplier_at_which_interaction_vanishes":
                 float(np.sqrt(v_within / 2) / sigma_z)})
    return rows


def transfer_regret(q: pd.DataFrame, sigma_z: float) -> dict:
    """What does the interaction cost a user who ranks designs in the wrong context?

    Take the design that measured better in c1, order it, and evaluate it in c2. The gap to
    the better design in c2 is the efficiency a perfectly context-aware chooser would have
    gained over a perfectly context-blind one, on exactly these matched pairs. This is the
    ceiling for any context-adaptation method on this data, and it is the number the
    plan's break-even analysis needs.
    """
    y1a, y1b = q.y_d1c1.to_numpy(), q.y_d2c1.to_numpy()
    y2a, y2b = q.y_d1c2.to_numpy(), q.y_d2c2.to_numpy()
    pick_from_c1 = np.where(y1a >= y1b, y2a, y2b)      # transfer c1's winner into c2
    best_c2 = np.maximum(y2a, y2b)
    rand_c2 = (y2a + y2b) / 2
    se = np.sqrt(2.0) * sigma_z
    big = (q.dz1.abs() > 2 * se) & (q.dz2.abs() > 2 * se)
    out = {}
    for name, m in (("all_quartets", np.ones(len(q), bool)),
                    ("supported_quartets", big.to_numpy())):
        if m.sum() == 0:
            continue
        out[name] = {
            "n": int(m.sum()),
            "oracle_efficiency_in_c2": float(best_c2[m].mean()),
            "transferred_choice_efficiency": float(pick_from_c1[m].mean()),
            "random_choice_efficiency": float(rand_c2[m].mean()),
            "transfer_regret": float((best_c2 - pick_from_c1)[m].mean()),
            "random_regret": float((best_c2 - rand_c2)[m].mean()),
            "share_of_random_regret_removed_by_transfer": float(
                1 - (best_c2 - pick_from_c1)[m].mean()
                / max((best_c2 - rand_c2)[m].mean(), 1e-12)),
        }
    return out


def bootstrap_headline(q: pd.DataFrame, sigma_z: float, seed: int,
                       n_boot: int = 400) -> dict:
    """Locus-clustered percentile intervals for the three numbers that carry the argument.

    Quartets are not independent: one measurement enters many of them, and a target site
    contributes a whole block. Resampling target sites is the right unit.
    """
    rng = np.random.default_rng(seed)
    cl = C.clusters_of(q.reset_index(drop=True), "target_name")
    keys = ["interaction_share", "sd_interaction", "reversal_rate_supported",
            "transfer_regret_supported"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
        s = q.iloc[idx]
        vc = variance_components(s, sigma_z)
        rs = reversal_stats(s, sigma_z)
        tr = transfer_regret(s, sigma_z)
        acc["interaction_share"].append(vc["interaction_share_of_delta_variance"])
        acc["sd_interaction"].append(vc["sd_interaction_estimate"])
        acc["reversal_rate_supported"].append(rs["reversal_rate_supported"])
        acc["transfer_regret_supported"].append(
            tr.get("supported_quartets", {}).get("transfer_regret", np.nan))
    out = {}
    for k, v in acc.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[k] = {"ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                  "n_valid": int(arr.size), "n_boot": n_boot,
                  "n_clusters": len(cl)}
    return out


def by_context_pair(q: pd.DataFrame, sigma_z: float) -> list[dict]:
    """Which kind of context change carries the interaction, if any."""
    def kind(c1: str, c2: str) -> str:
        a, b = c1.split("|"), c2.split("|")
        same_study, same_cell, same_pe = a[0] == b[0], a[1] == b[1], a[2] == b[2]
        if same_study and same_cell and not same_pe:
            return "editor only (PE2 vs PE4)"
        if same_study and same_pe and not same_cell:
            return "cell line only"
        if same_study:
            return "cell and editor"
        return "across source studies"
    q = q.assign(kind=[kind(a, b) for a, b in zip(q.c1, q.c2)])
    rows = []
    for k, s in q.groupby("kind"):
        vc = variance_components(s, sigma_z)
        rs = reversal_stats(s, sigma_z)
        rows.append({"context_change": k, "n_quartets": int(len(s)),
                     "sd_interaction": vc["sd_interaction_estimate"],
                     "sd_design_main": vc["sd_design_main_effect_estimate"],
                     "reversal_rate_all": rs["reversal_rate_all"],
                     "n_supported": rs["n_both_contrasts_significant"],
                     "reversal_rate_supported": rs["reversal_rate_supported"]})
    return sorted(rows, key=lambda r: -r["n_quartets"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--max-pairs-per-cell", type=int, default=6)
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = pd.read_parquet(C.require_manifest())
    d["z"] = arcsine(d.edited.to_numpy())
    nm = noise_model(d)

    # replicates are measurements, not candidates
    d = (d.groupby(["edit_key", "design_key", "context_key", "cond3", "spacer",
                    "target_name", "edit_type"], observed=True, as_index=False)
          .agg(edited=("edited", "mean"), z=("z", "mean"), n_rep=("edited", "size")))

    cached = C.cache_path("quartets_full")
    if cached.exists():
        # The quartet enumeration is the slow step (about eight minutes) and E04 consumes
        # the same table, so it is built once and cached.
        q = pd.read_parquet(cached)
        q.attrs["n_design_pairs_dropped_by_cap"] = -1
    else:
        q = build_quartets(d, args.max_pairs_per_cell, rng)
        q.to_parquet(cached, index=False)
    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "noise_model": nm,
           "support": {"n_quartets": int(len(q)),
                       "n_alleles": int(q.edit_key.nunique()),
                       "n_target_sites": int(q.target_name.nunique()),
                       "n_context_pairs": int((q.c1 + "->" + q.c2).nunique()),
                       "design_pairs_dropped_by_cap":
                           int(q.attrs["n_design_pairs_dropped_by_cap"]),
                       "max_pairs_per_allele": args.max_pairs_per_cell}}
    res["observed"] = {
        "mean_abs_D_z": float(q.D_z.abs().mean()),
        "sd_D_z": float(q.D_z.std()),
        "mean_abs_D_raw": float(q.D_y.abs().mean()),
        "sd_D_raw": float(q.D_y.std()),
        "quantiles_abs_D_z": {qq: float(q.D_z.abs().quantile(qq))
                              for qq in (0.5, 0.9, 0.99)},
    }
    res["variance_components"] = variance_components(q, nm["sigma_z"])
    res["reversals"] = reversal_stats(q, nm["sigma_z"])
    res["additive_null"] = additive_null(q, nm["sigma_z"], args.seed, args.n_sim)
    res["noise_sensitivity"] = noise_sensitivity(q, nm["sigma_z"])
    res["transfer_regret"] = transfer_regret(q, nm["sigma_z"])
    res["bootstrap"] = bootstrap_headline(q, nm["sigma_z"], args.seed, args.n_boot)
    res["by_context_change"] = by_context_pair(q, nm["sigma_z"])

    C.write_outputs("e03_matched_support", res, render(res))


def render(r: dict) -> str:
    nm, sup, ob = r["noise_model"], r["support"], r["observed"]
    vc, rev, an = r["variance_components"], r["reversals"], r["additive_null"]
    L = ["# E03 - design-by-context interaction on matched candidate sets\n",
         "Descriptive. The only fitted object is a two-way additive mean decomposition, "
         "used as a null.\n",
         "## Support\n",
         f"- **{sup['n_quartets']:,} quartets** over {sup['n_alleles']:,} canonical alleles "
         f"and {sup['n_target_sites']:,} target sites, on {sup['n_context_pairs']} ordered "
         "context pairs.\n"
         f"- A quartet is two alternative designs for one allele, both measured in both "
         f"contexts. Design pairs are capped at {sup['max_pairs_per_allele']} per allele "
         f"({sup['design_pairs_dropped_by_cap']:,} pairs dropped).\n",
         "## Measurement noise\n",
         f"- {nm['n_groups']} metadata-identical replicate groups, {nm['df']} degrees of "
         f"freedom, {nm['frac_groups_from_kim']:.0%} of them from Kim.\n"
         f"- pooled per-measurement SD: **{nm['sigma_z']:.4f}** on the arcsine-root scale "
         f"({nm['sigma_raw']:.4f} on the raw efficiency scale).\n"
         f"- a single design contrast therefore carries SE {rev['se_of_one_contrast']:.4f}, "
         f"and a quartet's D carries SE {2*nm['sigma_z']:.4f}, before any real effect.\n",
         "## What the observed contrasts contain\n",
         "| component of the design contrast Delta | variance | SD |\n|---|---:|---:|",
         f"| total spread of Delta across alleles, design pairs, contexts | {vc['var_delta_total']:.5f} | {np.sqrt(vc['var_delta_total']):.4f} |",
         f"| design-pair main effect | {vc['var_design_main_effect_estimate']:.5f} | {vc['sd_design_main_effect_estimate']:.4f} |",
         f"| design x context interaction | {vc['var_interaction_estimate']:.5f} | {vc['sd_interaction_estimate']:.4f} |",
         f"| measurement noise (2 sigma^2) | {vc['var_noise_in_delta']:.5f} | {np.sqrt(vc['var_noise_in_delta']):.4f} |",
         f"\nThe interaction accounts for **{vc['interaction_share_of_delta_variance']:.1%}** of "
         f"the variance of the design contrast, against "
         f"{vc['noise_share_of_delta_variance']:.1%} for measurement noise and the rest for "
         "the context-independent design effect. It is estimated from "
         f"{vc['n_design_pairs_with_2plus_contexts']:,} (allele, design pair) combinations seen "
         "in two or more contexts.\n",
         "## Order changes\n",
         "| quartet subset | n | reversal rate |\n|---|---:|---:|",
         f"| all quartets | {rev['n_quartets']:,} | {rev['reversal_rate_all']:.3f} |",
         f"| both contrasts > 2 SE | {rev['n_both_contrasts_significant']:,} | {rev['reversal_rate_supported']:.3f} |"]
    for thr in (0.02, 0.05, 0.10):
        L.append(f"| both > 2 SE and both raw gaps > {thr:.2f} | "
                 f"{rev[f'n_supported_and_gap_gt_{thr}']:,} | "
                 f"{rev[f'reversal_rate_supported_and_gap_gt_{thr}']:.3f} |")
    L.append("\n## The additive counterexample\n")
    L.append("Under any additive latent model D is exactly measurement error, so its SD is "
             f"2*sigma = {an['analytic_sd_D_z']:.4f} analytically. Reversal rates are "
             "simulated conditional on each quartet's own additive contrast estimate "
             f"({an['n_sim']} simulations), so the null carries the observed contrast sizes "
             "and no interaction:\n")
    L.append("| statistic | observed | additive null (mean) | null 95% range |\n|---|---:|---:|---|")
    L.append(f"| mean \\|D\\| (arcsine) | {ob['mean_abs_D_z']:.4f} | {an['mean_abs_D_z']['mean']:.4f} | "
             f"[{an['mean_abs_D_z']['p2.5']:.4f}, {an['mean_abs_D_z']['p97.5']:.4f}] |")
    L.append(f"| SD of D (arcsine) | {ob['sd_D_z']:.4f} | {an['sd_D_z']['mean']:.4f} | "
             f"[{an['sd_D_z']['p2.5']:.4f}, {an['sd_D_z']['p97.5']:.4f}] |")
    L.append(f"| reversal rate, all | {rev['reversal_rate_all']:.3f} | {an['reversal_rate_all']['mean']:.3f} | "
             f"[{an['reversal_rate_all']['p2.5']:.3f}, {an['reversal_rate_all']['p97.5']:.3f}] |")
    L.append(f"| reversal rate, both > 2 SE | {rev['reversal_rate_supported']:.3f} | "
             f"{an['reversal_rate_supported']['mean']:.3f} | "
             f"[{an['reversal_rate_supported']['p2.5']:.3f}, {an['reversal_rate_supported']['p97.5']:.3f}] |")
    tr, bs = r["transfer_regret"], r["bootstrap"]
    L.append("\n## What the interaction costs a context-blind chooser\n")
    L.append("Order the design that measured better in one context, then evaluate it in the "
             "other. Raw efficiency units.\n")
    L.append("| quartet subset | n | oracle in c2 | transferred choice | random choice | "
             "transfer regret | share of random regret removed |\n|---|---:|---:|---:|---:|---:|---:|")
    for k, v in tr.items():
        L.append(f"| {k} | {v['n']:,} | {v['oracle_efficiency_in_c2']:.4f} | "
                 f"{v['transferred_choice_efficiency']:.4f} | {v['random_choice_efficiency']:.4f} | "
                 f"{v['transfer_regret']:.4f} | "
                 f"{v['share_of_random_regret_removed_by_transfer']:.1%} |")
    L.append("\n## Locus-clustered intervals for the load-bearing numbers\n")
    L.append(f"Resampling {bs['interaction_share']['n_clusters']:,} target sites, "
             f"{bs['interaction_share']['n_boot']} draws.\n")
    L.append("| statistic | estimate | 95% CI |\n|---|---:|---|")
    L.append(f"| interaction share of Var(Delta) | {vc['interaction_share_of_delta_variance']:.3f} | "
             f"[{bs['interaction_share']['ci95'][0]:.3f}, {bs['interaction_share']['ci95'][1]:.3f}] |")
    L.append(f"| SD of interaction (arcsine) | {vc['sd_interaction_estimate']:.4f} | "
             f"[{bs['sd_interaction']['ci95'][0]:.4f}, {bs['sd_interaction']['ci95'][1]:.4f}] |")
    L.append(f"| supported reversal rate | {rev['reversal_rate_supported']:.4f} | "
             f"[{bs['reversal_rate_supported']['ci95'][0]:.4f}, {bs['reversal_rate_supported']['ci95'][1]:.4f}] |")
    L.append(f"| transfer regret, supported quartets | "
             f"{tr['supported_quartets']['transfer_regret']:.4f} | "
             f"[{bs['transfer_regret_supported']['ci95'][0]:.4f}, {bs['transfer_regret_supported']['ci95'][1]:.4f}] |")
    L.append("\n## How wrong could the noise estimate be?\n")
    L.append("| sigma multiplier | SD interaction | interaction share of Var(Delta) |\n|---|---:|---:|")
    for row in r["noise_sensitivity"]:
        if "sigma_multiplier" not in row:
            continue
        L.append(f"| {row['sigma_multiplier']:.1f}x | {row['sd_interaction_estimate']:.4f} | "
                 f"{row['interaction_share']:.1%} |")
    vanish = [x for x in r["noise_sensitivity"]
              if "sigma_multiplier_at_which_interaction_vanishes" in x][0]
    L.append(f"\nThe interaction estimate reaches zero only if the true per-measurement SD "
             f"is **{vanish['sigma_multiplier_at_which_interaction_vanishes']:.2f}x** the "
             "replicate-based estimate, i.e. if the 654 replicate groups understate the "
             "noise by that factor.\n")
    L.append("\n## Which context change carries it\n")
    L.append("| context change | quartets | SD interaction | SD design main | reversal rate | supported n | supported reversal rate |\n|---|---:|---:|---:|---:|---:|---:|")
    for s in r["by_context_change"]:
        sup_rr = "n/a" if not np.isfinite(s["reversal_rate_supported"]) else f"{s['reversal_rate_supported']:.3f}"
        L.append(f"| {s['context_change']} | {s['n_quartets']:,} | {s['sd_interaction']:.4f} | "
                 f"{s['sd_design_main']:.4f} | {s['reversal_rate_all']:.3f} | {s['n_supported']:,} | {sup_rr} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
