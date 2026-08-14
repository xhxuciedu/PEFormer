"""Experimental-context categorical vocabularies (task spec §17).

Each context field gets its own small vocabulary with an explicit UNK id for values
unseen at fit time (or genuinely missing), rather than imputing from outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

UNK = "<unk>"

CONTEXT_FIELDS: tuple[str, ...] = (
    "cell_type",
    "pe_type",
    "cas9_type",
    "cas9_pam",
    "scaffold_name",
    "motif",
    "source_study",
)


@dataclass
class ContextVocab:
    fields: tuple[str, ...] = CONTEXT_FIELDS
    vocabs: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, fields: tuple[str, ...] = CONTEXT_FIELDS) -> "ContextVocab":
        vocabs = {}
        for f_ in fields:
            values = sorted(df[f_].dropna().astype(str).unique().tolist())
            vocabs[f_] = [UNK] + values
        return cls(fields=fields, vocabs=vocabs)

    def sizes(self) -> dict[str, int]:
        return {f_: len(v) for f_, v in self.vocabs.items()}

    def encode_row(self, row: pd.Series) -> dict[str, int]:
        out = {}
        for f_ in self.fields:
            val = row.get(f_)
            vocab = self.vocabs[f_]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                out[f_] = 0
            else:
                s = str(val)
                out[f_] = vocab.index(s) if s in vocab else 0
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"fields": list(self.fields), "vocabs": self.vocabs}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "ContextVocab":
        d = json.loads(Path(path).read_text())
        return cls(fields=tuple(d["fields"]), vocabs=d["vocabs"])
