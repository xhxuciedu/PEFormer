"""Regression tests on the assembled corpus's row counts (task spec §40).

Skips if the processed data hasn't been built in this environment (these are
integration checks over real pipeline output, not pure unit tests).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

HSU_PATH = Path("data/processed/hsu2026_74769.parquet")
CORPUS_PATH = Path("data/processed/optiprime_full_297962.parquet")

pytestmark = pytest.mark.skipif(
    not HSU_PATH.exists() or not CORPUS_PATH.exists(),
    reason="processed data not built in this environment",
)


def test_hsu_extraction_has_exactly_74769_rows():
    df = pd.read_parquet(HSU_PATH)
    assert len(df) == 74_769


def test_hsu_efficiency_in_unit_range():
    df = pd.read_parquet(HSU_PATH)
    assert df["editing_efficiency"].between(0, 1).all()


def test_corpus_efficiency_in_unit_range():
    df = pd.read_parquet(CORPUS_PATH)
    assert df["edited"].between(0, 1).all()
    assert df["indel"].between(0, 1).all()


def test_corpus_no_duplicate_record_ids():
    df = pd.read_parquet(CORPUS_PATH)
    assert df["record_id"].is_unique


def test_corpus_covers_all_three_source_studies():
    df = pd.read_parquet(CORPUS_PATH)
    assert set(df["source_study"].unique()) == {"hsu2026", "deepprime", "pridict_pridict2"}
