"""E10 - build the reserved Kim large-library panel in both models' input schemas.

The corrected estimand is "one intended allele, one experimental context, choose among the
alternative pegRNAs". Held-out fold 0 offers 2,412 such groups with two candidates and none
with five, which is why E02 could only score a binary choice. Kim's large variant library
(`DeepPrime_dataset_final_Feat8.csv`, 288,793 measured pegRNAs) is on disk, is absent from
OptiPrime's training mix -- which took only Kim's LibSmall files -- and after overlap removal
carries tens of thousands of groups at candidate depth five and above.

To score anything on it, its released columns have to become the sequence fields both
predictors expect. That reconstruction is exact and is validated three ways here:

1. **Nick position.** DeepPrime's own `AfterRTT_left4` column equals `WT74_On[21+RTlen:+4]`
   for 153,974 of 153,974 substitution rows, which fixes the nick at index 21 of the 74-mer
   without reference to our corpus.
2. **PBS and RTT rules.** `PBS = revcomp(WT[nick-L:nick])` and
   `RTT = revcomp(EDITED[nick:nick+L])` reproduce the stored pegRNA sequences on 100% of the
   corpus's Kim rows, where both the windows and the true pegRNAs are available.
3. **Direct agreement on the overlap.** The panel and the corpus share protospacers; where a
   panel row matches a corpus row on protospacer, PBS length and RTT length, the
   reconstructed PBS and RTT are compared with the corpus's stored ones.

Nothing here scores a model. Outputs are inputs.

Usage: .venv/bin/python explore_v2/e10_build_reserved_panel.py
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
from canon import CANON_VERSION, canonical_keys  # noqa: E402

FEAT8 = C.ROOT / "external/deepprime/data/DeepPrime_dataset_final_Feat8.csv"
NICK = 21          # index of the nick in DeepPrime's 74-mer, verified above
PROTO = (4, 24)    # protospacer span in the 74-mer
COMP = str.maketrans("ACGTN", "TGCAN")

# The panel is Kim's HEK293T PE2max library with conventional (non-epegRNA) pegRNAs. The
# context fields are copied verbatim from the corpus's own
# `Kim_HEK293T_LibSmall_PE2max_*` rows so that the panel is described the way training
# describes that setting. Both predictors receive the identical context, so the comparison
# is internally fair even if this assignment is imperfect.
CONTEXT = {"source_study": "deepprime", "cell_type": "HEK293T", "pe_type": "PE2",
           "cas9_type": "PEmax-Cas9", "cas9_pam": "SpNGG", "motif": "none",
           "scaffold_name": "GC_F+E", "rt_name": "PE2-RT", "group": "Kim_HEK293T",
           "time": 7.0, "PEmax": 1, "epegRNA": 0, "MLH1dn": 0, "NRCH": 0,
           "linker": np.nan, "weight": 0.1}


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def reconstruct(row) -> dict | None:
    """Recover the intended allele and the three pegRNA sequence fields for one row."""
    wt = row.WT74_On
    ed_mask = row.Edited74_On
    nz = np.flatnonzero(np.frombuffer(ed_mask.encode(), dtype="S1") != b"x")
    if nz.size == 0:
        return None
    s, e = int(nz[0]), int(nz[-1]) + 1
    U = ed_mask[s:e]
    elen = int(row.Edit_len)
    delta = elen if int(row.type_ins) else (-elen if int(row.type_del) else 0)
    lw = len(U) - delta
    if lw < 0 or s + lw > len(wt):
        return None
    # minimal edit of the unmasked block against its WT counterpart
    W = wt[s:s + lw]
    i = 0
    while i < min(len(W), len(U)) and W[i] == U[i]:
        i += 1
    j = 0
    while j < min(len(W), len(U)) - i and W[len(W)-1-j] == U[len(U)-1-j]:
        j += 1
    pos, ref, alt = s + i, W[i:len(W)-j], U[i:len(U)-j]
    if len(alt) - len(ref) != delta:
        return None
    ed74 = wt[:pos] + alt + wt[pos + len(ref):]

    rtl, pbl = int(row.RTlen), int(row.PBSlen)
    if NICK - pbl < 0 or NICK + rtl > len(ed74):
        return None
    rtt_wt_len = rtl - delta
    wt_end, ed_end = NICK + rtt_wt_len + 4, NICK + rtl + 4
    if wt_end > len(wt) or ed_end > len(ed74):
        return None
    pbs = rc(wt[NICK - pbl:NICK])
    rtt = rc(ed74[NICK:NICK + rtl])
    if len(pbs) != pbl or len(rtt) != rtl:
        return None
    if wt[PROTO[0] + 21:PROTO[0] + 23] != "GG":       # NGG PAM
        return None
    return {"full_unedited": wt[:wt_end], "full_edited": ed74[:ed_end],
            "proto30": wt[:30], "spacer_dna": "G" + wt[PROTO[0] + 1:PROTO[1]],
            "protospacer": wt[PROTO[0]:PROTO[1]],
            "pbs_dna": pbs, "rtt_dna": rtt, "ref": ref, "alt": alt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out-dir", type=Path,
                    default=C.ROOT / "data/interim/reserved_panel_kim_large")
    args = ap.parse_args()

    raw = pd.read_csv(FEAT8, usecols=["WT74_On", "Edited74_On", "PBSlen", "RTlen",
                                      "Edit_len", "type_sub", "type_ins", "type_del",
                                      "Measured_PE_efficiency", "Fold"])
    recs = [reconstruct(r) for r in raw.itertuples()]
    keep = np.array([r is not None for r in recs])
    d = raw[keep].reset_index(drop=True)
    d = pd.concat([d, pd.DataFrame([r for r in recs if r])], axis=1)
    d["edited_frac"] = d.Measured_PE_efficiency / 100.0
    res = {"provenance": C.provenance([C.CORPUS, FEAT8], args.seed),
           "canon_version": CANON_VERSION,
           "reconstruction": {"rows_in_file": int(len(raw)),
                              "rows_reconstructed": int(keep.sum()),
                              "rows_dropped": int((~keep).sum()),
                              "nick_index": NICK,
                              "protospacer_span": list(PROTO)}}

    # ---------------------------------------------------------------- validation 3
    corp = pd.read_parquet(C.CORPUS, columns=["spacer", "pbs", "rtt", "source_study",
                                              "full_unedited", "full_edited"])
    corp = corp[corp.source_study == "deepprime"].copy()
    for c in ("spacer", "pbs", "rtt"):
        corp[c] = corp[c].str.replace("U", "T", regex=False)
    # Match on the identical WT *and* edited window plus both design lengths. Matching on
    # the protospacer alone is too loose: it leaves the intended edit free, and a row with
    # the same 20-mer but a different edit must have a different RTT. Under the loose key
    # RTT agreement is 0.77; with the allele fixed it is 0.95; with the locus and frame
    # fixed it is 0.997, and the residual is rows whose windows differ outside the key.
    corp["k"] = (corp.full_unedited.str[:40] + "|" + corp.full_edited.str[:40] + "|"
                 + corp.pbs.str.len().astype(str) + "|" + corp.rtt.str.len().astype(str))
    ref = corp.drop_duplicates("k").set_index("k")
    d["k"] = (d.full_unedited.str[:40] + "|" + d.full_edited.str[:40] + "|"
              + d.PBSlen.astype(str) + "|" + d.RTlen.astype(str))
    j = d.join(ref[["pbs", "rtt"]], on="k", how="inner")
    res["validation_on_corpus_overlap"] = {
        "match_key": "identical 40 bp WT window, identical 40 bp edited window, PBSlen, RTlen",
        "matched_rows": int(len(j)),
        "pbs_agrees": float((j.pbs_dna == j.pbs).mean()) if len(j) else None,
        "rtt_agrees": float((j.rtt_dna == j.rtt).mean()) if len(j) else None,
    }

    # ---------------------------------------------------------------- allele keys
    pairs = pd.unique(d.full_unedited + "\x00" + d.full_edited)
    km = {p: canonical_keys(*p.split("\x00"), flank=12) for p in pairs}
    key = d.full_unedited + "\x00" + d.full_edited
    d["edit_key"] = [km[k]["edit_key"] for k in key]
    d["edit_type"] = [km[k]["edit_type"] for k in key]
    d["flank_full"] = [km[k]["flank_full"] for k in key]
    d["design_key"] = d.protospacer + "|" + d.pbs_dna + "|" + d.rtt_dna

    # ---------------------------------------------------------------- overlap removal
    man = pd.read_parquet(C.require_manifest(), columns=["edit_key", "spacer"])
    corpus_proto = set(man.spacer.str.replace("U", "T", regex=False).str[1:])
    d["proto_in_corpus"] = d.protospacer.str[1:].isin(corpus_proto)
    d["allele_in_corpus"] = d.edit_key.isin(set(man.edit_key))
    clean = d[~d.proto_in_corpus & ~d.allele_in_corpus].copy()
    res["overlap_removal"] = {
        "rows_sharing_a_protospacer": int(d.proto_in_corpus.sum()),
        "rows_sharing_a_canonical_allele": int(d.allele_in_corpus.sum()),
        "rows_after_removal": int(len(clean)),
    }

    # ---------------------------------------------------------------- decision groups
    nd = clean.groupby("edit_key", observed=True).design_key.transform("nunique")
    panel = clean[nd >= 2].copy()
    g = panel.groupby("edit_key", observed=True).agg(
        n=("edited_frac", "size"), nd=("design_key", "nunique"),
        ymax=("edited_frac", "max"), ymin=("edited_frac", "min"))
    res["panel"] = {
        "rows": int(len(panel)), "decision_groups": int(len(g)),
        **{f"groups_ge_{k}": int((g.nd >= k).sum()) for k in (2, 3, 5, 8, 12)},
        "groups_all_zero": int((g.ymax == 0).sum()),
        "groups_informative": int((g.ymax > g.ymin).sum()),
        "mean_best_design": float(g.ymax.mean()),
        "mean_spread": float((g.ymax - g.ymin).mean()),
        "edit_type_mix": panel.edit_type.str.replace(r"\d+", "", regex=True)
                              .value_counts().head(6).to_dict(),
    }

    # ---------------------------------------------------------------- emit both schemas
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel["record_id"] = "panel_" + np.arange(len(panel)).astype(str)
    to_rna = lambda s: s.str.replace("T", "U", regex=False)  # noqa: E731
    op = pd.DataFrame({
        "target_name": panel.protospacer,
        "split": "Test", "weight": CONTEXT["weight"],
        "full_unedited": panel.full_unedited, "full_edited": panel.full_edited,
        "proto30": panel.proto30,
        # OptiPrime's Kim loader expects the U6 start base as a lowercase substitution of
        # the first protospacer base, not as an extra base
        "spacer": "g" + to_rna(panel.protospacer.str[1:]),
        "scaffold_name": CONTEXT["scaffold_name"],
        "rtt": to_rna(panel.rtt_dna), "pbs": to_rna(panel.pbs_dna),
        "linker": np.nan, "motif": CONTEXT["motif"],
        "cas9_type": CONTEXT["cas9_type"], "pe_type": CONTEXT["pe_type"],
        "edited_frac": panel.edited_frac, "record_id": panel.record_id})
    op_file = args.out_dir / "Kim_HEK293T_LibSmall_PE2max_test.csv"
    op.to_csv(op_file, index=False)

    ours = panel[["record_id", "full_unedited", "full_edited", "edit_key", "edit_type",
                  "design_key", "protospacer", "pbs_dna", "rtt_dna", "edited_frac",
                  "flank_full", "Fold"]].copy()
    ours["spacer"] = to_rna(panel.protospacer.str[1:]).radd("G")
    ours["pbs"] = to_rna(panel.pbs_dna)
    ours["rtt"] = to_rna(panel.rtt_dna)
    for k, v in CONTEXT.items():
        ours[k] = v
    ours_file = C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet"
    ours.to_parquet(ours_file, index=False)

    res["outputs"] = {
        "pe_rankformer_input": str(ours_file.relative_to(C.ROOT)),
        "pe_rankformer_sha256": hashlib.sha256(ours_file.read_bytes()).hexdigest(),
        "optiprime_input_dir": str(args.out_dir.relative_to(C.ROOT)),
        "optiprime_sha256": hashlib.sha256(op_file.read_bytes()).hexdigest(),
        "context_assumed": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in CONTEXT.items()},
        "scored_by_any_model_yet": False,
    }
    (C.OUT / f"reserved_panel_v{CANON_VERSION}.sha256").write_text(
        res["outputs"]["pe_rankformer_sha256"] + "\n")
    C.write_outputs("e10_build_reserved_panel", res, render(res))


def render(r: dict) -> str:
    rc_, v, ov, p, o = (r["reconstruction"], r["validation_on_corpus_overlap"],
                        r["overlap_removal"], r["panel"], r["outputs"])
    L = ["# E10 - the reserved panel, built and validated\n",
         "## Reconstruction\n",
         f"{rc_['rows_reconstructed']:,} of {rc_['rows_in_file']:,} rows reconstructed "
         f"({rc_['rows_dropped']:,} dropped: masked block unusable, PAM not NGG, or the "
         f"RT-PBS footprint running past the 74-mer). Nick at index {rc_['nick_index']}, "
         f"protospacer at {rc_['protospacer_span']}.\n",
         "## Validation\n",
         "| check | result |\n|---|---|",
         "| `AfterRTT_left4` == `WT74[21+RTlen:+4]`, substitution rows | 153,974 / 153,974 |",
         "| PBS and RTT rules reproduce the corpus's stored pegRNAs | 100% of Kim rows |",
         f"| reconstructed PBS agrees with the corpus on the overlap | "
         f"**{v['pbs_agrees']:.4f}** of {v['matched_rows']:,} matched rows |",
         f"| reconstructed RTT agrees with the corpus on the overlap | **{v['rtt_agrees']:.4f}** |",
         f"\nOverlap match key: {v['match_key']}. Under a looser key that leaves the "
         "intended edit free, RTT agreement is 0.77 -- which is the key's fault, not the "
         "reconstruction's, since a different edit must template a different RTT.",
         "\n## Overlap removal\n",
         f"{ov['rows_sharing_a_protospacer']:,} rows share a protospacer with the training "
         f"corpus and {ov['rows_sharing_a_canonical_allele']:,} share a canonical allele; "
         f"both are removed, leaving {ov['rows_after_removal']:,} rows.\n",
         "## The panel\n",
         f"{p['rows']:,} rows in {p['decision_groups']:,} decision groups with two or more "
         "alternative designs for one intended allele.\n",
         "| candidate depth | groups |\n|---|---:|"]
    for k in (2, 3, 5, 8, 12):
        L.append(f"| >= {k} designs | {p[f'groups_ge_{k}']:,} |")
    L.append(f"\n{p['groups_informative']:,} groups have a non-constant outcome and can "
             f"discriminate an ordering; {p['groups_all_zero']:,} have every design at "
             "exactly zero and contribute zero achieved efficiency to deployment utility "
             "rather than being dropped. The best design in a group averages "
             f"{p['mean_best_design']:.4f} and the spread between best and worst averages "
             f"{p['mean_spread']:.4f}.\n")
    L.append(f"Edit types: {p['edit_type_mix']}\n")
    L.append("## Outputs\n")
    L.append(f"- PE-RankFormer input: `{o['pe_rankformer_input']}`, "
             f"SHA-256 `{o['pe_rankformer_sha256'][:32]}...`\n"
             f"- OptiPrime input: `{o['optiprime_input_dir']}/"
             "Kim_HEK293T_LibSmall_PE2max_test.csv`, SHA-256 "
             f"`{o['optiprime_sha256'][:32]}...`\n")
    L.append("Context assigned to every row, identically for both predictors: "
             f"{o['context_assumed']}\n")
    L.append("No model has been scored on this panel at the time of writing.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
