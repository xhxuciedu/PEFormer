"""Outcome-independent decision eligibility from acquired supplementary sequences."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from common import ROOT, OUT, provenance, write_json
from canon import canonical_keys

RAW = OUT / "data/raw"
SCAFFOLD = "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"

def shortname(name):
    m = re.match(r"^(\d+)[_-]?(NM|TD)", str(name))
    return "".join(m.groups()) if m else "unmapped"

def main():
    source_path = ROOT / "explore_v2/cache/corpus_canonical_v2.parquet"
    source = pd.read_parquet(source_path, columns=["edit_key","spacer","source_study","cell_type","pe_type"])
    source_alleles, source_core19 = set(source.edit_key), set(source.spacer.str[-19:])
    oped = {}
    p = RAW / "oped_clinvar.xlsx"
    for sheet in ("Table S3", "PE tools", "top 3 designs", "NG PAM"):
        df = pd.read_excel(p, sheet_name=sheet, header=1)
        df.columns = df.columns.str.strip()
        allele = "ClinVarAlleleID" if "ClinVarAlleleID" in df else "ClinVar AlleleID"
        df[allele] = df[allele].ffill()
        depth = df.groupby(allele).size()
        oped[sheet] = {"design_rows": len(df), "alleles": len(depth),
                       "max_design_rows_per_allele": int(depth.max()),
                       "measured_efficiency_columns": [], "columns": list(df.columns),
                       "status": "designs only; cannot evaluate efficiency without linked outcomes"}
    ep = pd.read_excel(RAW / "epridict_supplements.xlsx", sheet_name="Arrayed_Editing_Results_PE")
    batches = [pd.read_csv(RAW / f"epridict_{kind}_batch.txt", sep="\t") for kind in ("highlow","additional")]
    seq = pd.concat(batches, ignore_index=True)
    seq = seq.loc[~seq.name.str.contains("control", case=False)].copy()
    seq["short"] = seq.name.map(shortname)
    seq = seq.drop_duplicates(["short","amplicon_seq","expected_hdr_amplicon_seq"])
    keys = {}
    for r in seq.itertuples():
        wt, ed = str(r.amplicon_seq).upper(), str(r.expected_hdr_amplicon_seq).upper()
        if not set(wt+ed) <= set("ACGT") or wt == ed:
            continue
        info = canonical_keys(wt, ed)
        keys.setdefault(r.short, {})[info["edit_key"]] = info
    records = []
    for r in ep.itertuples():
        name = shortname(r.pegRNA_shortname)
        options = keys.get(name, {})
        guide = str(r.pegRNA_sequence).upper()
        pieces = guide.split(SCAFFOLD)
        spacer = pieces[0] if len(pieces) == 2 else ""
        allele = next(iter(options)) if len(options) == 1 else None
        records.append({"shortname":r.pegRNA_shortname,"sequence_key":name,"allele_options":len(options),
            "edit_key":allele,"spacer":spacer,"guide":guide,
            "chromosome":r.chromosome,"position":int(r.position),"edit_class":r.Correction_Type,
            "source_allele_overlap": allele in source_alleles if allele else None,
            "source_spacer_core19_overlap": spacer[-19:] in source_core19 if spacer else None,
            "peg_tokens_including_special":len(pieces[0])+len(pieces[1])+2 if len(pieces)==2 else None,
            "has_K562_outcome":pd.notna(r.K562_edited_percentage_endogenous),
            "has_HEK293T_outcome":pd.notna(r.HEK293T_edited_percentage_endogenous)})
    f = pd.DataFrame(records)
    details = {}
    for cell in ("K562","HEK293T"):
        sub = f.loc[f[f"has_{cell}_outcome"]]
        mapped = sub.loc[sub.edit_key.notna()]
        clean = mapped.loc[(mapped.source_allele_overlap == False) &
                           (mapped.source_spacer_core19_overlap == False)]
        for label, block in (("mapped",mapped),("sequence_overlap_excluded",clean)):
            depth = block.groupby("edit_key").guide.nunique()
            details[f"{cell}_{label}"] = {"rows":len(block),"alleles":len(depth),
                "groups_ge2":int((depth>=2).sum()),"groups_ge5":int((depth>=5).sum()),
                "groups_ge8":int((depth>=8).sum()),"max_depth":int(depth.max()) if len(depth) else 0}
    f.to_parquet(OUT / "data/epridict_eligibility.parquet", index=False)
    result = {"inputs_sha256":provenance([source_path,p,RAW/"epridict_supplements.xlsx",
               RAW/"epridict_highlow_batch.txt",RAW/"epridict_additional_batch.txt",__file__]),
        "oped":oped,"epridict":{"rows":len(f),"unambiguous_alleles":int(f.edit_key.notna().sum()),
            "ambiguous_rows":int((f.allele_options>1).sum()),"unmapped_rows":int((f.allele_options==0).sum()),
            "recognized_scaffold_rows":int((f.spacer.str.len()>0).sum()),"contexts":details},
        "source_contexts":source.groupby(["source_study","cell_type","pe_type"]).size().rename("rows").reset_index().to_dict("records"),
        "confirmation_status":"Not established: OPED outcomes missing; see actual ePRIDICT fixed-allele depths",
        "no_external_model_scoring":True,
        "limitations":["Core-19 spacer exclusion is conservative, not full homology clustering",
           "ePRIDICT is an independent experiment within a source-adjacent study family",
           "Some displayed spreadsheet examples were inspected for schema; no model-outcome comparisons performed",
           "No outcome values used to select decision groups or resolve allele identities"]}
    write_json(OUT/"data_inventory.json",result)
    lines = ["# V3-01: acquired-data eligibility audit", "", "No model was scored on either acquired panel.", "",
        "## OPED", "", "The supplement has 30 primary ClinVar designs, 40 tool-comparison designs over 8 alleles,",
        "and 18 top-three designs over 6 alleles. These sheets have no measured-efficiency columns.",
        "The [primary paper](https://www.nature.com/articles/s42256-023-00739-w) points to raw sequencing",
        "under PRJNA882795; linking and processing it is still needed. Do not invent outcomes from design ranks.",
        "", "## ePRIDICT", "", f"Acquired {len(f)} arrayed pegRNA rows and authors' CRISPResso sequence manifests.",
        f"Resolved {f.edit_key.notna().sum()} rows to an unambiguous canonical intended allele;",
        f"{(f.allele_options>1).sum()} ambiguous and {(f.allele_options==0).sum()} unmapped rows remain excluded.",
        "Grouping uses intended edited versus unedited amplicon sequences, not only locus/edit class.", "",
        "| Population | Rows | Alleles | Depth >=2 | >=5 | >=8 | Maximum |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,d in details.items():
        lines.append(f"| {name} | {d['rows']} | {d['alleles']} | {d['groups_ge2']} | {d['groups_ge5']} | {d['groups_ge8']} | {d['max_depth']} |")
    lines += ["", "Exact allele and conservative spacer-core overlaps are excluded in the second population.",
        "This does not establish full homology independence. Editor/context and input support still require",
        "validation before inference. Published data: [ePRIDICT supplementary files](https://github.com/Schwank-Lab/epridict/tree/supplementary_files).", "",
        "## Gate decision", "", "Two independent deep-selection studies are not yet secured. Broad independent confirmation",
        "is **not ready**. Continue only the bounded retrospective Kim development screen and protocol",
        "work; defer an expanded architecture sweep and strong generalization claims.", "",
        "Hashes, source-context counts, per-design eligibility and unresolved rows are recorded in",
        "`data_inventory.json` and `data/epridict_eligibility.parquet`. No external outcomes were used for tuning."]
    (OUT/"DATA_AUDIT.md").write_text("\n".join(lines)+"\n")
    print(result["epridict"], flush=True)

if __name__ == "__main__":
    main()
