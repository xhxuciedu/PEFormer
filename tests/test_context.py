"""Context vocabulary encoding (task spec section 40)."""

from __future__ import annotations

import pandas as pd

from pe_rankformer.data.context import ContextVocab


def _toy_df():
    return pd.DataFrame(
        {
            "cell_type": ["HEK293T", "HeLa", "HEK293T", None],
            "pe_type": ["PE2", "PE4", "PE2", "PE2"],
        }
    )


def test_fit_includes_unk_at_index_zero():
    vocab = ContextVocab.fit(_toy_df(), fields=("cell_type", "pe_type"))
    assert vocab.vocabs["cell_type"][0] == "<unk>"


def test_missing_value_encodes_to_unk():
    vocab = ContextVocab.fit(_toy_df(), fields=("cell_type", "pe_type"))
    row = pd.Series({"cell_type": None, "pe_type": "PE2"})
    enc = vocab.encode_row(row)
    assert enc["cell_type"] == 0


def test_unseen_value_encodes_to_unk():
    vocab = ContextVocab.fit(_toy_df(), fields=("cell_type", "pe_type"))
    row = pd.Series({"cell_type": "K562", "pe_type": "PE2"})
    enc = vocab.encode_row(row)
    assert enc["cell_type"] == 0


def test_known_value_roundtrips():
    vocab = ContextVocab.fit(_toy_df(), fields=("cell_type", "pe_type"))
    row = pd.Series({"cell_type": "HeLa", "pe_type": "PE4"})
    enc = vocab.encode_row(row)
    assert vocab.vocabs["cell_type"][enc["cell_type"]] == "HeLa"
    assert vocab.vocabs["pe_type"][enc["pe_type"]] == "PE4"


def test_save_load_roundtrip(tmp_path):
    vocab = ContextVocab.fit(_toy_df(), fields=("cell_type", "pe_type"))
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = ContextVocab.load(path)
    assert loaded.vocabs == vocab.vocabs
    assert loaded.fields == vocab.fields
