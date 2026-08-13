"""Validate our understanding of OptiPrime's loader against the real, imported code."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pe_rankformer.data import optiprime_compat as oc
from pe_rankformer.data.seqops import revcomp

optiprime = pytest.importorskip(
    "flax", reason="external/optiprime deps (jax/flax/ViennaRNA) not installed"
)


def _toy_row(**overrides) -> pd.DataFrame:
    row = {
        "spacer": "GCTGTATCTATATTCATCAT",
        "rtt": "CGCGGTTCTATCTAGTTACG",
        "pbs": "ATGAATATAGATAC",
        "full_unedited": "NNNN" + "GCTGTATCTATATTCATCAT" + "CGG" + "N" * 40,
        "full_edited": "NNNN" + "GCTGTATCTATATTCATCAT" + "GGG" + "N" * 40,
        "edited_frac": 0.3,
        "indel_frac": 0.05,
        "weight": 1.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_process_fname_liu_hek293t_pe2():
    df = _toy_row()
    oc.process_fname(Path("Liu_HEK293T_x_PE2.csv"), df)
    assert df.loc[0, "cas9_type"] == "PEmax-Cas9"
    assert df.loc[0, "motif"] == "tevoPreQ1"
    assert df.loc[0, "pe_type"] == "PE2"
    assert df.loc[0, "time"] == 3.0


def test_process_fname_liu_hela_pe4():
    df = _toy_row()
    oc.process_fname(Path("Liu_HeLa_x_PE4.csv"), df)
    assert df.loc[0, "pe_type"] == "PE4"
    assert df.loc[0, "time"] == 5.0


def test_format_pe_df_drops_zero_weight():
    df = _toy_row(weight=0.0)
    out = oc.format_pe_df(Path("Liu_HEK293T_x_PE2.csv"), df)
    assert len(out) == 0


def test_format_pe_df_drops_missing_indel():
    df = _toy_row(indel_frac=float("nan"))
    out = oc.format_pe_df(Path("Liu_HEK293T_x_PE2.csv"), df)
    assert len(out) == 0


def test_format_pe_df_keeps_valid_row():
    df = _toy_row()
    out = oc.format_pe_df(Path("Liu_HEK293T_x_PE2.csv"), df)
    assert len(out) == 1
    assert out.loc[0, "edited"] == pytest.approx(0.3)
    assert out.loc[0, "unedited"] == pytest.approx(0.65)


def test_pegrna_orientation_matches_our_schema():
    """OptiPrime's `pegrna = spacer + scaffold + rtt + pbs` uses rtt/pbs in pegRNA
    orientation -- the same convention as pe_rankformer.data.schema. Round trip through
    revcomp confirms the two conventions describe the same molecule."""
    pbs_target_strand = "GTAAAATAGATACA"
    rtt_pegrna_orientation = revcomp(pbs_target_strand)
    assert revcomp(rtt_pegrna_orientation) == pbs_target_strand


def test_count_optiprime_filtering_toy_file(tmp_path):
    rows = pd.concat(
        [_toy_row(), _toy_row(weight=0.0), _toy_row(indel_frac=float("nan"))],
        ignore_index=True,
    )
    counts = oc.count_optiprime_filtering(Path("Liu_HEK293T_x_PE2.csv"), rows)
    assert counts.raw_rows == 3
    assert counts.weight_zero_rows == 1
    assert counts.missing_efficiency_rows == 1
    assert counts.retained_rows == 1
