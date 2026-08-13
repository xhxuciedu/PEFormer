"""Parse the Hsu et al. 2026 Supplementary workbook into the canonical long format.

Two library sheets (Lib-MMR, Lib-CV) each carry four editing/indel column pairs, one
per experimental context (HEK293T/HeLa x PE2/PE4). Melting them and dropping the
missing editing values yields the 74,769 measurements used by OptiPrime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import CANONICAL_COLUMNS
from .seqops import diff_window, find_protospacer, revcomp

logger = logging.getLogger(__name__)

SHEET_LIBMMR = "Supp Table 4 LibMMR"
SHEET_LIBCV = "Supp Table 5 LibCV"
SHEET_ENDO = "Supp Table 3 Endo_gRNAs"

EXPECTED_TOTAL = 74_769

# Nick site sits 3 nt 5' of the PAM, i.e. after the 17th protospacer base.
NICK_OFFSET_IN_PROTOSPACER = 17

CONTEXTS: tuple[tuple[str, str, str], ...] = (
    # (column prefix, cell_type, prime_editor)
    ("HEK293T_PE2", "HEK293T", "PE2"),
    ("HEK293T_PE4", "HEK293T", "PE4"),
    ("HeLa_PE2", "HeLa", "PE2"),
    ("HeLa_PE4", "HeLa", "PE4"),
)

# PE4 is PE2 plus a dominant-negative MLH1; the workbook encodes nothing else about
# the condition, so this is a definition rather than an inference.
PE_CONDITION = {"PE2": "MLH1dn_absent", "PE4": "MLH1dn_present"}


@dataclass(frozen=True, slots=True)
class SheetCounts:
    """Per-context nonmissing efficiency counts for one sheet."""

    dataset: str
    n_designs: int
    per_context: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.per_context.values())


def _designs_libmmr(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Lib-MMR design columns to the canonical per-design fields."""
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = "Lib-MMR"
    out["source_row_id"] = df["ID"].astype(str)
    out["target_id"] = "Lib-MMR:" + df["Name, spacer-target"].astype(str)
    out["design_name"] = df["Name"].astype(str)
    out["design_category"] = df["Design category"].astype(str)
    out["spacer"] = df["Designed 5G pegRNA spacer"].str.upper()
    out["unedited_sequence"] = df["Designed target (ps-pam-edit)"].str.upper()
    out["edited_sequence"] = df["Designed edited target (ps-pam-edit)"].str.upper()

    pbs_target_strand = df["PBS"].str.upper()
    homology_arm = df["Homology arm"].str.upper()
    extension = df["Designed pegRNA extension (pbs-edit-hom)"].str.upper()
    # extension reads PBS + edit product + homology arm on the protospacer strand.
    edit_product = [
        ext[len(pbs) : len(ext) - len(hom)]
        for ext, pbs, hom in zip(extension, pbs_target_strand, homology_arm)
    ]
    out["pbs_sequence"] = pbs_target_strand.map(revcomp)
    out["rtt_sequence"] = [revcomp(e + h) for e, h in zip(edit_product, homology_arm)]
    return out


def _designs_libcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Lib-CV design columns to the canonical per-design fields."""
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = "Lib-CV"
    out["source_row_id"] = df["index"].astype(str)
    out["target_id"] = "Lib-CV:" + df["mutation_name"].astype(str)
    out["design_name"] = df["gene"].astype(str)
    out["design_category"] = "ClinVar"
    out["spacer"] = df["spacer"].str.upper()
    out["unedited_sequence"] = df["unedited_target"].str.upper()
    out["edited_sequence"] = df["edited_target"].str.upper()

    pbs_target_strand = df["pbs_bind"].str.upper()
    homology_arm = df["homology_arm"].str.upper()
    edit_product = df["edit_product"].fillna("").astype(str).str.upper()
    out["pbs_sequence"] = pbs_target_strand.map(revcomp)
    out["rtt_sequence"] = [revcomp(e + h) for e, h in zip(edit_product, homology_arm)]
    return out


def _annotate_designs(designs: pd.DataFrame) -> pd.DataFrame:
    """Add protospacer/PAM location and edit description derived from the sequences."""
    protospacers: list[str | None] = []
    pams: list[str | None] = []
    nick_positions: list[int | None] = []

    for target, spacer in zip(designs["unedited_sequence"], designs["spacer"]):
        hit = find_protospacer(target, spacer)
        if hit is None:
            protospacers.append(None)
            pams.append(None)
            nick_positions.append(None)
            continue
        start, proto = hit
        end = start + len(proto)
        pam = target[end : end + 3]
        protospacers.append(proto)
        pams.append(pam if len(pam) == 3 else None)
        nick_positions.append(start + NICK_OFFSET_IN_PROTOSPACER)

    designs["protospacer"] = protospacers
    designs["pam"] = pams

    specs = [
        diff_window(wt, ed)
        for wt, ed in zip(designs["unedited_sequence"], designs["edited_sequence"])
    ]
    designs["edit_type"] = [s.edit_type for s in specs]
    designs["edit_length"] = [s.edit_length for s in specs]
    designs["edit_position"] = [s.start for s in specs]
    designs["edit_ref"] = [s.ref for s in specs]
    designs["edit_alt"] = [s.alt for s in specs]
    designs["edit_position_from_nick"] = [
        None if nick is None else s.start - nick for s, nick in zip(specs, nick_positions)
    ]

    designs["pbs_length"] = designs["pbs_sequence"].str.len()
    designs["rtt_length"] = designs["rtt_sequence"].str.len()
    return designs


def _melt_contexts(designs: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Expand one design table into one row per (design, experimental context)."""
    frames = []
    for prefix, cell_type, editor in CONTEXTS:
        eff = pd.to_numeric(raw[f"{prefix}_editing"], errors="coerce")
        indel = pd.to_numeric(raw[f"{prefix}_indel"], errors="coerce")
        keep = eff.notna()
        block = designs.loc[keep].copy()
        block["editing_efficiency"] = eff.loc[keep].to_numpy()
        block["indel_rate"] = indel.loc[keep].to_numpy()
        block["cell_type"] = cell_type
        block["prime_editor"] = editor
        block["pe_condition"] = PE_CONDITION[editor]
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def count_efficiencies(workbook: Path) -> list[SheetCounts]:
    """Count nonmissing editing measurements per sheet and context, without parsing."""
    counts = []
    for sheet, dataset in ((SHEET_LIBMMR, "Lib-MMR"), (SHEET_LIBCV, "Lib-CV")):
        raw = pd.read_excel(workbook, sheet_name=sheet)
        per_context = {
            prefix: int(pd.to_numeric(raw[f"{prefix}_editing"], errors="coerce").notna().sum())
            for prefix, _, _ in CONTEXTS
        }
        counts.append(SheetCounts(dataset, len(raw), per_context))
    return counts


def load_hsu2026(workbook: Path) -> pd.DataFrame:
    """Build the canonical long-format table for the Hsu et al. 2026 libraries."""
    frames = []
    for sheet, normalizer in (
        (SHEET_LIBMMR, _designs_libmmr),
        (SHEET_LIBCV, _designs_libcv),
    ):
        raw = pd.read_excel(workbook, sheet_name=sheet)
        designs = _annotate_designs(normalizer(raw))
        logger.info("%s: %d designs", sheet, len(designs))
        frames.append(_melt_contexts(designs, raw))

    table = pd.concat(frames, ignore_index=True)
    table["source_study"] = "hsu2026"
    table["scaffold_type"] = pd.NA
    table["epegRNA_flag"] = pd.NA
    table["experimental_context_id"] = pd.NA
    table["record_id"] = (
        "hsu2026:"
        + table["source_dataset"]
        + ":"
        + table["source_row_id"]
        + ":"
        + table["cell_type"]
        + "_"
        + table["prime_editor"]
    )

    extra = ["design_name", "design_category"]
    return table[CANONICAL_COLUMNS + extra]
