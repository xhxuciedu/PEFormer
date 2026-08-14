"""Sequence alignment and edit classification (task spec §40)."""

from __future__ import annotations

from pe_rankformer.data.seqops import align_pair, diff_window, find_protospacer, revcomp


def test_revcomp_basic():
    assert revcomp("ACGT") == "ACGT"
    assert revcomp("AAGG") == "CCTT"


def test_revcomp_roundtrip():
    seq = "ACGTACGTGGCCTTAA"
    assert revcomp(revcomp(seq)) == seq


def test_diff_window_substitution():
    spec = diff_window("ACGCTA", "ACGTTA")
    assert spec.edit_type == "substitution"
    assert spec.ref == "C"
    assert spec.alt == "T"
    assert spec.start == 3


def test_diff_window_insertion():
    spec = diff_window("ACGTA", "ACGAATA")
    assert spec.edit_type == "insertion"
    assert spec.ref == ""


def test_diff_window_deletion():
    spec = diff_window("ACGAATA", "ACGTA")
    assert spec.edit_type == "deletion"
    assert spec.alt == ""


def test_diff_window_identical_sequences():
    spec = diff_window("ACGTACGT", "ACGTACGT")
    assert spec.edit_type == "none"
    assert spec.edit_length == 0


def test_diff_window_multi_substitution():
    spec = diff_window("AACCGGTT", "AAGCGCTT")
    assert spec.edit_type == "multi_substitution"
    assert spec.n_mismatch == 2


def test_align_pair_equal_length_positions_match():
    a, b = align_pair("ACGCTA", "ACGTTA")
    assert len(a) == len(b) == 6
    assert a[3] == "C" and b[3] == "T"


def test_align_pair_insertion_introduces_gap_in_wt():
    wt, edited = align_pair("ACGTA", "ACGAATA")
    assert len(wt) == len(edited)
    assert "-" in wt


def test_align_pair_deletion_introduces_gap_in_edited():
    wt, edited = align_pair("ACGAATA", "ACGTA")
    assert len(wt) == len(edited)
    assert "-" in edited


def test_find_protospacer_exact_match():
    target = "TTTTACGTACGTACGTACGTACGTGGG"
    hit = find_protospacer(target, "ACGTACGTACGTACGTACGT")
    assert hit is not None
    idx, proto = hit
    assert target[idx : idx + len(proto)] == proto


def test_find_protospacer_5g_trim():
    genomic = "CGTACGTACGTACGTACGT"
    target = "TTTT" + genomic + "TGGG"
    hit = find_protospacer(target, "G" + genomic)  # non-genomic 5' G
    assert hit is not None
    idx, proto = hit
    assert proto == genomic


def test_find_protospacer_prefers_leftmost_tandem_duplicate():
    genomic = "ACGTACGTACGTACGTACGT"
    target = genomic + "NNNNNNNNNN" + genomic  # tandem duplication
    hit = find_protospacer(target, genomic)
    assert hit is not None
    idx, _ = hit
    assert idx == 0


def test_find_protospacer_not_found():
    assert find_protospacer("ACGTACGT", "TTTTTTTTTTTTTTTTTTTT") is None
