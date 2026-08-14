"""Paired-token encoding and pegRNA segmentation (task spec section 40)."""

from __future__ import annotations

from pe_rankformer.data.tokenizer import (
    EDIT_PAD_ID,
    EDIT_VOCAB_SIZE,
    NUC_PAD_ID,
    NUC_VOCAB_SIZE,
    N_SEGMENTS,
    SEGMENT_TO_ID,
    encode_edit_pair,
    encode_pegrna,
)


def test_edit_vocab_size_25_states_plus_specials():
    # 5 bases (incl gap) x 5 bases = 25, plus pad/bos/eos
    assert EDIT_VOCAB_SIZE == 25 + 3


def test_encode_edit_pair_substitution():
    ids, length = encode_edit_pair("ACGCTA", "ACGTTA", max_len=20)
    assert length == 6 + 2  # BOS + 6 positions + EOS
    assert ids[0] != EDIT_PAD_ID  # BOS present
    assert ids[length:].count(EDIT_PAD_ID) == len(ids) - length


def test_encode_edit_pair_identical_sequence_has_diagonal_tokens():
    ids, length = encode_edit_pair("ACGT", "ACGT", max_len=10)
    # every non-special token should represent (base, base) i.e. same base twice
    assert length == 4 + 2


def test_encode_edit_pair_truncates_long_sequences():
    wt = "A" * 500
    ed = "A" * 499 + "C"
    ids, length = encode_edit_pair(wt, ed, max_len=50)
    assert length <= 52
    assert len(ids) == 52


def test_encode_pegrna_segments_present():
    enc = encode_pegrna("ACGTACGTAC", "GGCCTTAA", "TTTTCCCCAAAA", max_len=40)
    assert enc.segment_ids[0] == SEGMENT_TO_ID["<pad>"]  # BOS position
    seg_values = set(enc.segment_ids[1 : enc.length - 1])
    assert seg_values == {SEGMENT_TO_ID["SPACER"], SEGMENT_TO_ID["PBS"], SEGMENT_TO_ID["RTT"]}


def test_encode_pegrna_length_matches_concatenation():
    spacer, pbs, rtt = "ACGT" * 5, "GG" * 6, "TT" * 8
    enc = encode_pegrna(spacer, pbs, rtt, max_len=100)
    expected = len(spacer) + len(pbs) + len(rtt) + 2  # + BOS/EOS
    assert enc.length == expected


def test_encode_pegrna_pads_to_max_len():
    enc = encode_pegrna("ACGT", "GG", "TT", max_len=30)
    assert len(enc.nuc_ids) == 30
    assert len(enc.segment_ids) == 30
    assert enc.nuc_ids[-1] == NUC_PAD_ID


def test_nuc_vocab_covers_acgtn():
    assert NUC_VOCAB_SIZE == 3 + 5 + 1  # specials + ACGT+gap + N


def test_n_segments_matches_spacer_pbs_rtt_plus_pad():
    assert N_SEGMENTS == 4
