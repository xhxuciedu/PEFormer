"""Vocabularies and tokenization for PE-RankFormer.

Two token streams per example (task spec §13-14):

1. Paired WT/edit tokens: each aligned position becomes one of 25 (base, base) states
   plus gap combinations, so substitutions/insertions/deletions are directly visible.
2. pegRNA tokens: single-nucleotide spacer/PBS/RTT with a segment-type embedding
   (handled by the model, not this module -- this module just emits per-position
   nucleotide ids and segment ids).

Context (cell type, editor, etc.) is handled separately in `context.py` since it's a
small set of categorical scalars, not a sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .seqops import GAP, align_pair

# ---- Paired edit-token vocabulary --------------------------------------------------
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
_BASES = ("A", "C", "G", "T", GAP)
EDIT_SPECIALS = (PAD, BOS, EOS)
EDIT_VOCAB: list[str] = list(EDIT_SPECIALS) + [f"{a}{b}" for a in _BASES for b in _BASES]
EDIT_TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(EDIT_VOCAB)}
EDIT_PAD_ID = EDIT_TOKEN_TO_ID[PAD]
EDIT_VOCAB_SIZE = len(EDIT_VOCAB)

# ---- pegRNA nucleotide vocabulary ---------------------------------------------------
NUC_VOCAB: list[str] = [PAD, BOS, EOS] + list(_BASES) + ["N"]
NUC_TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(NUC_VOCAB)}
NUC_PAD_ID = NUC_TOKEN_TO_ID[PAD]
NUC_VOCAB_SIZE = len(NUC_VOCAB)

# Segment types for the pegRNA stream.
SEGMENTS = (PAD, "SPACER", "PBS", "RTT")
SEGMENT_TO_ID: dict[str, int] = {seg: i for i, seg in enumerate(SEGMENTS)}
SEGMENT_PAD_ID = SEGMENT_TO_ID[PAD]
N_SEGMENTS = len(SEGMENTS)


def encode_edit_pair(wt: str, edited: str, max_len: int) -> tuple[list[int], int]:
    """Align WT/edited sequences and encode as paired-base token ids.

    Returns (ids, length) where ids is BOS + tokens + EOS, right-padded to max_len + 2.
    Truncates from the center-out if the aligned pair exceeds max_len (rare; long
    indels), keeping the start and end so region-boundary information isn't lost.
    """
    a, b = align_pair(wt.upper(), edited.upper())
    tokens = [f"{x}{y}" for x, y in zip(a, b)]
    if len(tokens) > max_len:
        keep = max_len
        head = keep // 2
        tail = keep - head
        tokens = tokens[:head] + tokens[-tail:]
    ids = [EDIT_TOKEN_TO_ID[BOS]]
    ids += [EDIT_TOKEN_TO_ID.get(t, EDIT_TOKEN_TO_ID[BOS]) for t in tokens]
    ids.append(EDIT_TOKEN_TO_ID[EOS])
    length = len(ids)
    ids += [EDIT_PAD_ID] * (max_len + 2 - length)
    return ids, length


@dataclass(frozen=True, slots=True)
class PegRNAEncoding:
    nuc_ids: list[int]
    segment_ids: list[int]
    length: int


def encode_pegrna(spacer: str, pbs: str, rtt: str, max_len: int) -> PegRNAEncoding:
    """Concatenate spacer/PBS/RTT (each uppercased DNA) into one nucleotide stream
    with per-position segment ids, framed by BOS/EOS."""
    parts = [(spacer.upper(), "SPACER"), (pbs.upper(), "PBS"), (rtt.upper(), "RTT")]
    nuc_ids = [NUC_TOKEN_TO_ID[BOS]]
    seg_ids = [SEGMENT_PAD_ID]
    for seq, seg in parts:
        seg_id = SEGMENT_TO_ID[seg]
        for base in seq:
            b = base if base in ("A", "C", "G", "T") else "N"
            nuc_ids.append(NUC_TOKEN_TO_ID[b])
            seg_ids.append(seg_id)
    nuc_ids.append(NUC_TOKEN_TO_ID[EOS])
    seg_ids.append(SEGMENT_PAD_ID)

    if len(nuc_ids) > max_len:
        nuc_ids = nuc_ids[: max_len - 1] + [NUC_TOKEN_TO_ID[EOS]]
        seg_ids = seg_ids[: max_len - 1] + [SEGMENT_PAD_ID]
    length = len(nuc_ids)
    pad_n = max_len - length
    nuc_ids = nuc_ids + [NUC_PAD_ID] * pad_n
    seg_ids = seg_ids + [SEGMENT_PAD_ID] * pad_n
    return PegRNAEncoding(nuc_ids=nuc_ids, segment_ids=seg_ids, length=length)
