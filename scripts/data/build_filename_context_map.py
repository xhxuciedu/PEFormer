"""Construct and validate OptiPrime-compatible filenames for all 42 dataset partitions.

For each partition in data/manifests/optiprime_context_counts.csv, build a filename that
follows OptiPrime's `{lab}_{cell_type}_{...}_{details}.csv` convention (recovered in
reports/optiprime_data_loader_reverse_engineering.md), then run the *actual* imported
`process_fname` on it and verify the resulting cas9_type/pe_type/motif/cas9_pam match the
partition's published PEmax/epegRNA/MLH1dn/NRCH flags. This both documents the filename
convention (data_collect_prompt.md section D) and produces the exact filenames later used
to build OptiPrime-compatible CSVs (section T).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data import optiprime_compat as oc  # noqa: E402

MANIFEST = Path("data/manifests/optiprime_context_counts.csv")
OUT = Path("reports/optiprime_filename_context_map.csv")

GENERAL_RULES = """\
# OptiPrime filename convention (decoded from external/optiprime/scripts/pe/pe_datasets.py,
# validated by running the real process_fname on every constructed filename below).
#
# All files: `{lab_prefix}_{name_parts[1]}_{name_parts[2]}_{name_parts[3]}.csv`, dispatched by
# lab_prefix = stem.split('_', 1)[0] in process_fname().
#
# Liu      (this study):        name_parts[1]=cell_type in {HEK293T,HeLa}; name_parts[3]=pe_type in {PE2,PE4}.
#                                Fixed: scaffold=BlpI_F+E, motif=tevoPreQ1, cas9_type=PEmax-Cas9.
#                                time = 3.0 (HEK293T) or 5.0 (HeLa). epegRNA/NRCH not filename-derived (always epegRNA here).
# Schwank  (PRIDICT/PRIDICT2.0): name_parts[3]=type_editor; pe_type=type_editor[:3], PEmax iff type_editor[3:]=='max'.
#                                cas9_type = PEmax-Cas9 if PEmax else PE2-Cas9. time = 7.0 fixed.
#                                epegRNA/NRCH are NOT set from filename -- Schwank source rows must carry their own
#                                'motif' column since epegRNA status varies within some partitions (e.g. HEK293T bars 5-8).
# Kim      (DeepPrime):         name_parts[2]=lib_name (time=8.0 iff 'LibClinvar' else 7.0); name_parts[3]=details.
#                                pe_type=details[:3]; PEmax iff 'max' in details; epegRNA iff '-e' in details (hyphen
#                                required -- 'PE4maxe' without hyphen does NOT set epegRNA); NRCH iff 'NRCH' in details.
# YKim     (PAM-variant, unused in the released weights -- see loader report section 4):
#                                name_parts[2]=PAM variant string used verbatim as cas9_pam. cas9_type=PE2-Cas9,
#                                pe_type=PE2, motif=none, time=3.0 fixed.
"""


def liu_filename(cell_type: str, mlh1dn: int) -> str:
    pe = "PE4" if mlh1dn else "PE2"
    return f"Liu_{cell_type}_Lib_{pe}.csv"


def schwank_filename(cell_type: str, pemax: int, mlh1dn: int) -> str:
    pe = "PE4" if mlh1dn else "PE2"
    suffix = "max" if pemax else ""
    return f"Schwank_{cell_type}_Lib_{pe}{suffix}.csv"


def kim_filename(cell_type: str, pemax: int, epegrna: int, mlh1dn: int, nrch: int) -> str:
    pe = "PE4" if mlh1dn else "PE2"
    details = pe + ("max" if pemax else "")
    if epegrna:
        details += "-e"
    if nrch:
        details += "NRCH"
    return f"Kim_{cell_type}_LibVariant_{details}.csv"


def build_filename(row: pd.Series) -> str:
    if row.lab == "Liu":
        return liu_filename(row.cell_type, row.MLH1dn)
    if row.lab == "Schwank":
        return schwank_filename(row.cell_type, row.PEmax, row.MLH1dn)
    if row.lab == "Kim":
        return kim_filename(row.cell_type, row.PEmax, row.epegRNA, row.MLH1dn, row.NRCH)
    raise ValueError(row.lab)


def validate(filename: str, row: pd.Series) -> tuple[bool, str]:
    df = pd.DataFrame({"x": [1]})
    p = Path(filename)
    oc.process_fname(p, df)

    expect_pe = "PE4" if row.MLH1dn else "PE2"
    if df.loc[0, "pe_type"] != expect_pe:
        return False, f"pe_type {df.loc[0, 'pe_type']} != {expect_pe}"

    if row.lab in ("Liu", "Schwank", "Kim"):
        expect_max = bool(row.PEmax)
        got_max = df.loc[0, "cas9_type"] == "PEmax-Cas9"
        if got_max != expect_max:
            return False, f"PEmax mismatch: {df.loc[0, 'cas9_type']}"

    if row.lab == "Kim":
        expect_epeg = bool(row.epegRNA)
        got_epeg = df.loc[0, "motif"] == "tevoPreQ1"
        if got_epeg != expect_epeg:
            return False, f"epegRNA mismatch: motif={df.loc[0, 'motif']}"
        expect_nrch = bool(row.NRCH)
        got_nrch = df.loc[0, "cas9_pam"] == "SpNRCH"
        if got_nrch != expect_nrch:
            return False, f"NRCH mismatch: cas9_pam={df.loc[0, 'cas9_pam']}"

    return True, "ok"


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    rows = []
    all_ok = True
    for _, r in manifest.iterrows():
        fname = build_filename(r)
        ok, msg = validate(fname, r)
        all_ok &= ok
        rows.append(
            {
                "bar_index": r.bar_index,
                "lab_prefix": r.lab,
                "cell_type": r.cell_type,
                "constructed_filename": fname,
                "PEmax": r.PEmax,
                "epegRNA": r.epegRNA,
                "MLH1dn": r.MLH1dn,
                "NRCH": r.NRCH,
                "n_target": r.n,
                "validated_against_process_fname": ok,
                "validation_note": msg,
            }
        )
    out_df = pd.DataFrame(rows)
    OUT.write_text(GENERAL_RULES + "\n" + out_df.to_csv(index=False))
    print(f"wrote {OUT}: {len(out_df)} rows, all_validated={all_ok}")
    assert all_ok, "some constructed filenames did not validate against process_fname"


if __name__ == "__main__":
    main()
