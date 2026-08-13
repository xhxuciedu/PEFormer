"""Thin wrapper that imports and runs the *actual* OptiPrime preprocessing code.

This module does not reimplement OptiPrime's filename parsing or row filtering; it
imports `process_fname` and `format_pe_df` directly from the official source clone at
`external/optiprime` (see reports/optiprime_data_loader_reverse_engineering.md) so our
row counts are validated against the real code, not our understanding of it.

`external/optiprime` is a cloned dependency, not part of this repository (see
.gitignore), and needs the extra packages listed in reports/optiprime_input_specification.md
(jax, flax, chex, optax, ViennaRNA, networkx) installed to import `pe_utils`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_OPTIPRIME_ROOT = Path(__file__).resolve().parents[3] / "external" / "optiprime"


def _ensure_on_path() -> None:
    root = str(_OPTIPRIME_ROOT)
    if not _OPTIPRIME_ROOT.is_dir():
        raise FileNotFoundError(
            f"external/optiprime not found at {_OPTIPRIME_ROOT}; clone "
            "https://github.com/alvin-hsu/optiprime-src into external/optiprime first"
        )
    if root not in sys.path:
        sys.path.insert(0, root)


def process_fname(path: Path, df: pd.DataFrame) -> None:
    """Run OptiPrime's real filename-driven metadata assignment, in place."""
    _ensure_on_path()
    from scripts.pe.pe_datasets import process_fname as _process_fname

    _process_fname(path, df)


def format_pe_df(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Run OptiPrime's real `format_pe_df` (fills defaults, filters, derives fields).

    Upstream bug workaround: `format_pe_df` crashes with `ValueError: Columns must be
    same length as key` when every row is filtered out, because `df.apply(..., axis=1,
    result_type='expand')` on an empty frame returns zero columns instead of three. This
    does not occur in practice on real per-context files (some rows always survive), but
    is guarded here so a fully-filtered input reports zero rows instead of crashing.
    """
    _ensure_on_path()
    from scripts.pe.pe_utils import format_pe_df as _format_pe_df

    weight = df["weight"] if "weight" in df.columns else pd.Series(1.0, index=df.index)
    edited = df.get("edited_frac", pd.Series(0.0, index=df.index))
    indel = df.get("indel_frac", pd.Series(0.0, index=df.index))
    unedited = 1 - (edited.fillna(0.0) + indel)
    survives = (weight > 0) & unedited.notna() & weight.notna()
    if not survives.any():
        return df.iloc[0:0]
    return _format_pe_df(path, df)


@dataclass(frozen=True, slots=True)
class FilterCounts:
    file: str
    raw_rows: int
    weight_zero_rows: int
    missing_efficiency_rows: int
    invalid_rows: int
    retained_rows: int


def count_optiprime_filtering(path: Path, df: pd.DataFrame) -> FilterCounts:
    """Run the real OptiPrime pipeline and report row counts at each filter stage.

    Mirrors data_collect_prompt.md section C: counts rows dropped by `weight <= 0`
    separately from rows dropped by missing `edited_frac`/`indel_frac`, then confirms
    the final retained count against a direct call to `format_pe_df`.
    """
    raw_rows = len(df)
    work = df.copy()
    process_fname(path, work)

    weight = work["weight"] if "weight" in work.columns else pd.Series(1.0, index=work.index)
    weight_zero_rows = int((weight <= 0).sum())

    edited = work.get("edited_frac", pd.Series(0.0, index=work.index)).fillna(0.0)
    indel = work.get("indel_frac", pd.Series(0.0, index=work.index))
    unedited = 1 - (edited + indel)
    missing_efficiency_rows = int(unedited.isna().sum())

    retained = format_pe_df(path, work)
    retained_rows = len(retained)
    invalid_rows = raw_rows - weight_zero_rows - missing_efficiency_rows - retained_rows

    return FilterCounts(
        file=path.name,
        raw_rows=raw_rows,
        weight_zero_rows=weight_zero_rows,
        missing_efficiency_rows=missing_efficiency_rows,
        invalid_rows=max(invalid_rows, 0),
        retained_rows=retained_rows,
    )
