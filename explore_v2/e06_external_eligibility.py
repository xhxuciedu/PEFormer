"""E06 - what external evaluation surface actually exists, and what must be acquired.

Research plan section 4 and section 8A days 8-14: "identify and reserve external panels
through metadata inspection". Two separate jobs:

1. **Audit what is already on disk.** The repository ships the four upstream method
   repositories and several raw supplements. Some of those files contain measured rows
   that are *not* in OptiPrime's 318,471-row training mix. Those are the cheapest
   untouched surfaces available, and one of them turns out to be the only place in this
   project with real candidate depth for a fixed-allele choice.

2. **Write the acquisition queue** for the genuinely independent candidates, with the
   per-file manifest fields the plan requires, marked as not yet acquired.

This script deliberately does **not** score any model on the reserved panel. The plan is
explicit that a panel used to choose a model becomes development data, and the panel is
worth more sealed than spent.

Usage: .venv/bin/python explore_v2/e06_external_eligibility.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import edit_type, left_normalise, minimal_edit, revcomp  # noqa: E402

FEAT8 = C.ROOT / "external/deepprime/data/DeepPrime_dataset_final_Feat8.csv"
ENDO = C.ROOT / "data/raw/hsu2026/41587_2026_3261_MOESM3_ESM.xlsx"


# --------------------------------------------------------------------------- #
# Kim's large library: reconstruct the intended allele from a masked edited window
# --------------------------------------------------------------------------- #
def feat8_allele(wt: str, ed: str, is_ins: bool, is_del: bool, elen: int,
                 flank: int = 12) -> dict:
    """Recover the intended allele from DeepPrime's masked `Edited74_On`.

    That column shows the edited sequence only inside the RT-PBS footprint and masks the
    rest with 'x', so the raw string pair is not comparable across designs at all: three
    designs for one 1 bp deletion produce three different masked strings. The unmasked
    block is aligned back onto the WT window using the row's own edit type and length, and
    the allele is then keyed exactly as in canon.py, which makes it comparable with the
    training corpus.
    """
    nz = np.flatnonzero(np.frombuffer(ed.encode(), dtype="S1") != b"x")
    if nz.size == 0:
        return {}
    s, e = int(nz[0]), int(nz[-1]) + 1
    U = ed[s:e]
    delta = elen if is_ins else (-elen if is_del else 0)
    lw = len(U) - delta
    if lw < 0 or s + lw > len(wt):
        return {}
    W = wt[s:s + lw]
    pos, ref, alt = minimal_edit(W, U)
    pos += s
    pos, ref, alt = left_normalise(wt, pos, ref, alt)
    left = wt[max(0, pos - flank):pos]
    right = wt[pos + len(ref):pos + len(ref) + flank]
    fwd = f"{left}|{ref}>{alt}|{right}"
    rev = f"{revcomp(right)}|{revcomp(ref)}>{revcomp(alt)}|{revcomp(left)}"
    return {"edit_key": min(fwd, rev), "ref": ref, "alt": alt,
            "edit_type": edit_type(ref, alt), "edit_pos": pos,
            "flank_full": int(len(left) == flank and len(right) == flank)}


def audit_feat8(man: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    f = pd.read_csv(FEAT8, usecols=["WT74_On", "Edited74_On", "PBSlen", "RTlen",
                                    "Edit_pos", "Edit_len", "type_sub", "type_ins",
                                    "type_del", "Measured_PE_efficiency", "Fold"])
    recs = [feat8_allele(w, e, bool(i), bool(d), int(L)) for w, e, i, d, L in
            zip(f.WT74_On, f.Edited74_On, f.type_ins, f.type_del, f.Edit_len)]
    ok = np.array([bool(r) for r in recs])
    f = f[ok].reset_index(drop=True)
    add = pd.DataFrame([r for r in recs if r])
    f = pd.concat([f, add], axis=1)
    # DeepPrime reports percentages; the corpus stores fractions
    f["edited"] = f.Measured_PE_efficiency / 100.0
    f["design_key"] = (f.WT74_On.str[4:24] + "|" + f.PBSlen.astype(str) + "|"
                       + f.RTlen.astype(str))
    f["spacer_dna"] = f.WT74_On.str[4:24]

    corpus_spacers = set(man.spacer.str.replace("U", "T", regex=False))
    corpus_alleles = set(man.edit_key)
    f["spacer_in_corpus"] = f.spacer_dna.isin(corpus_spacers)
    f["allele_in_corpus"] = f.edit_key.isin(corpus_alleles)

    g = f.groupby("edit_key", observed=True).agg(
        n_design=("design_key", "nunique"), n=("edited", "size"),
        y_max=("edited", "max"), y_min=("edited", "min"),
        overlap=("allele_in_corpus", "max"))
    clean = f[~f.allele_in_corpus & ~f.spacer_in_corpus]
    gc = clean.groupby("edit_key", observed=True).agg(
        n_design=("design_key", "nunique"), n=("edited", "size"),
        y_max=("edited", "max"), y_min=("edited", "min"))
    res = {
        "file": str(FEAT8.relative_to(C.ROOT)),
        "rows_in_file": int(len(ok)), "rows_parsed": int(ok.sum()),
        "unique_wt_windows": int(f.WT74_On.nunique()),
        "canonical_alleles": int(f.edit_key.nunique()),
        "designs": int(f.design_key.nunique()),
        "edit_type_mix": f.edit_type.str.replace(r"\d+", "", regex=True)
                          .value_counts().head(6).to_dict(),
        "efficiency_units": "per cent in file, divided by 100 here",
        "mean_efficiency": float(f.edited.mean()),
        "frac_exact_zero": float((f.edited == 0).mean()),
        "candidate_depth": {f"alleles_ge_{k}": int((g.n_design >= k).sum())
                            for k in (2, 3, 5, 8)},
        "overlap_with_training_corpus": {
            "rows_sharing_a_protospacer": int(f.spacer_in_corpus.sum()),
            "rows_sharing_a_canonical_allele": int(f.allele_in_corpus.sum()),
            "alleles_sharing_a_canonical_allele": int(g.overlap.sum()),
        },
        "after_removing_any_overlap": {
            "rows": int(len(clean)), "alleles": int(clean.edit_key.nunique()),
            **{f"alleles_ge_{k}": int((gc.n_design >= k).sum()) for k in (2, 3, 5, 8)},
            "mean_within_allele_range_ge2": float(
                (gc.y_max - gc.y_min)[gc.n_design >= 2].mean()),
            "mean_best_design_ge2": float(gc.y_max[gc.n_design >= 2].mean()),
        },
    }
    return res, clean


def audit_endo() -> dict:
    d = pd.read_excel(ENDO, sheet_name="Supp Table 3 Endo_gRNAs")
    sp = "(e)pegRNA spacer"
    rtt = "(e)pegRNA RTT"
    pbs = "(e)pegRNA PBS"
    g = d.groupby(sp).agg(n=(rtt, "size"), n_rtt=(rtt, "nunique"), n_pbs=(pbs, "nunique"))
    return {"file": str(ENDO.relative_to(C.ROOT)),
            "sheet": "Supp Table 3 Endo_gRNAs",
            "rows": int(len(d)), "columns": list(map(str, d.columns)),
            "distinct_spacers": int(d[sp].nunique()),
            "spacers_with_more_than_one_design": int((g.n > 1).sum()),
            "max_designs_per_spacer": int(g.n.max()),
            "median_designs_per_spacer": float(g.n.median()),
            "editors": d["Editor"].value_counts().to_dict(),
            "has_efficiency_column": False,
            "verdict": ("Confirms the plan's inventory: 283 endogenous design rows with "
                        "spacer, scaffold, RTT, PBS, motif and nicking guide, and no "
                        "measured outcome in this sheet. It is a design panel, not an "
                        "evaluation panel, until per-figure source data are recovered.")}


# Every "blocking" entry marked AUDITED was measured in this program; the rest are read
# from primary publication and data-availability descriptions and remain unverified.
ACQUISITION = [
    {"candidate": "MinsePIE / Koeppel et al.", "status": "not acquired",
     "why": "repair-dependent insertion behaviour; read-count tables released",
     "blocking": "pooled insertion rates are library-abundance normalised, not per-design "
                 "edited-allele fractions; long inserts may exceed the 90-token pegRNA input",
     "fields_to_recover": "accession, replicate, cell line, MMR status, insert sequence, "
                          "read counts, library abundance denominator"},
    {"candidate": "OPED original prospective experiments (PRJNA882795)",
     "status": "not acquired",
     "why": "original endogenous measurements from a different group; the plan's first "
            "choice for external selection",
     "blocking": "usable row counts require supplement extraction; restrict to the "
                 "single-pegRNA configuration this model supports",
     "fields_to_recover": "guide, target window, cell line, editor, replicate, efficiency "
                          "denominator"},
    {"candidate": "PRIDICT2 / ePRIDICT endogenous and chromatin panels (PRJNA1025026)",
     "status": "partially on disk",
     "why": "assay-shift test; endogenous rather than reporter",
     "blocking": "the same source family already contributes 174,067 training rows, so "
                 "only panels proven absent from training qualify; "
                 "external/pridict2/dataset holds the 23k processed library and a "
                 "ranking-percentile table, both of which need an overlap audit against "
                 "the training mix before use",
     "fields_to_recover": "endogenous vs mapped-reporter flag, integration site, "
                          "coordinates, replicate identity"},
    {"candidate": "Li et al. chromatin/repair experiments (PRJNA949965)",
     "status": "not acquired",
     "why": "position and chromatin-state effects at mapped integration sites",
     "blocking": "few shared reporter designs, so probably cannot support alternative-"
                 "pegRNA selection; mechanistic validation only",
     "fields_to_recover": "integration site, chromatin annotation provenance, perturbation"},
    {"candidate": "OptiPrime/Hsu endogenous panel (Supp Table 3)",
     "status": "on disk, no outcomes",
     "why": "matched-design endogenous panel from a training-adjacent source",
     "blocking": "no efficiency column in the released sheet; needs per-figure source data",
     "fields_to_recover": "measured efficiency per design per figure panel, replicate, cell"},
    {"candidate": "StopPR functional screen", "status": "not acquired",
     "why": "downstream application validation",
     "blocking": "readout is guide abundance/fitness, not editing efficiency; do not use "
                 "as an efficiency label",
     "fields_to_recover": "phenotype readout definition, genotype confirmation subset"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    man = pd.read_parquet(C.require_manifest(),
                          columns=["edit_key", "spacer", "fold"])
    res = {"provenance": C.provenance([C.CORPUS, FEAT8], args.seed)}
    res["kim_large_library"], clean = audit_feat8(man)
    res["optiprime_endogenous_panel"] = audit_endo()
    res["acquisition_queue"] = ACQUISITION

    # NOTE: E10 supersedes this block. It rebuilds the panel from exact design sequences
    # with a three-way validated reconstruction, and it is what E11 scores. The file written
    # here uses a coarse (protospacer, PBS length, RT length) design key and is kept only as
    # the eligibility survey that motivated E10.
    # reserve the panel: write it once, hash it, and record that no model has seen it
    keep = clean.groupby("edit_key", observed=True).design_key.transform("nunique") >= 2
    panel = clean[keep][["WT74_On", "Edited74_On", "PBSlen", "RTlen", "edit_key",
                         "edit_type", "design_key", "spacer_dna", "edited", "Fold"]]
    out = C.OUT / "reserved_panel_kim_large.parquet"
    panel.to_parquet(out, index=False)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    res["reserved_panel"] = {
        "path": str(out.relative_to(C.ROOT)), "sha256": h,
        "rows": int(len(panel)), "alleles": int(panel.edit_key.nunique()),
        "decision_groups_ge2": int(panel.edit_key.nunique()),
        "scored_by_any_model_in_this_program": False,
        "seal_note": ("Written and hashed without any model being run on it. The first "
                      "evaluation on this panel must be a pre-declared comparison; any "
                      "earlier use turns it into development data."),
    }
    (C.OUT / "reserved_panel_kim_large.sha256").write_text(h + "\n")
    C.write_outputs("e06_external_eligibility", res, render(res))


def render(r: dict) -> str:
    k, e = r["kim_large_library"], r["optiprime_endogenous_panel"]
    p = r["reserved_panel"]
    L = ["# E06 - external eligibility, and one reserved panel\n",
         "## 1. Kim's large library is on disk and is not in the training mix\n",
         f"`{k['file']}` holds {k['rows_parsed']:,} measured rows over "
         f"{k['unique_wt_windows']:,} WT windows, {k['canonical_alleles']:,} canonical "
         f"alleles and {k['designs']:,} distinct designs. Mean efficiency "
         f"{k['mean_efficiency']:.4f}, {k['frac_exact_zero']:.1%} exact zeros.\n",
         "The released `Edited74_On` column is masked outside the RT-PBS footprint, so the "
         "raw string pair cannot identify an allele: three designs for one deletion give "
         "three different masked strings. Aligning the unmasked block back onto the WT "
         "window with each row's own edit type recovers the allele and makes it comparable "
         "with the training corpus.\n",
         "| candidate depth | alleles |\n|---|---:|"]
    for kk, v in k["candidate_depth"].items():
        L.append(f"| {kk.replace('alleles_ge_', '>= ')} designs | {v:,} |")
    ov = k["overlap_with_training_corpus"]
    L.append(f"\nOverlap with the 318,471-row training mix: "
             f"{ov['rows_sharing_a_protospacer']:,} rows share a protospacer and "
             f"{ov['rows_sharing_a_canonical_allele']:,} rows share a canonical allele "
             f"({ov['alleles_sharing_a_canonical_allele']:,} alleles). Removing every row "
             "with either kind of overlap leaves:\n")
    a = k["after_removing_any_overlap"]
    L.append(f"- {a['rows']:,} rows, {a['alleles']:,} alleles\n"
             f"- **{a['alleles_ge_2']:,} decision groups with two or more alternative "
             f"designs**, {a['alleles_ge_3']:,} with three or more, {a['alleles_ge_5']:,} "
             f"with five or more\n"
             f"- within those groups the best design averages {a['mean_best_design_ge2']:.4f} "
             f"and the spread between best and worst averages "
             f"{a['mean_within_allele_range_ge2']:.4f}\n")
    L.append("For comparison, the manuscript's held-out fold 0 offers 2,412 two-design "
             "groups and none with five. This panel is the same laboratory and the same "
             "assay, so it is **not** an independent-study test; it is an untouched surface "
             "with the candidate depth the corrected decision estimand needs.\n")
    L.append("## 2. The OptiPrime/Hsu endogenous panel has designs but no outcomes\n")
    L.append(f"`{e['file']}`, sheet `{e['sheet']}`: {e['rows']} rows over "
             f"{e['distinct_spacers']} spacers, median {e['median_designs_per_spacer']:.0f} "
             f"designs per spacer, editors {e['editors']}. No efficiency column.\n")
    L.append(e["verdict"] + "\n")
    L.append("## 3. Acquisition queue\n")
    L.append("| candidate | status | why it is wanted | what blocks it |\n|---|---|---|---|")
    for c in r["acquisition_queue"]:
        L.append(f"| {c['candidate']} | {c['status']} | {c['why']} | {c['blocking']} |")
    L.append("\nNothing in this queue has been downloaded, normalised or scored in this "
             "program. Published library sizes are not counts of usable evaluation rows.\n")
    L.append("## 4. The reserved panel\n")
    L.append(f"`{p['path']}` - {p['rows']:,} rows, {p['alleles']:,} decision groups with at "
             f"least two candidates.\nSHA-256 `{p['sha256'][:32]}...`\n")
    L.append(p["seal_note"] + "\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
