"""Sequence utilities: reverse complement, WT/edited alignment, edit classification.

All DNA is handled uppercase in the protospacer-strand 5'->3' orientation unless a
function name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

GAP: Final[str] = "-"
_COMPLEMENT: Final[dict[str, str]] = {
    "A": "T", "C": "G", "G": "C", "T": "A", "N": "N", GAP: GAP,
}


def revcomp(seq: str) -> str:
    """Reverse complement. Raises on characters outside ACGTN and gap."""
    return "".join(_COMPLEMENT[b] for b in reversed(seq.upper()))


@dataclass(frozen=True, slots=True)
class EditSpec:
    """A single contiguous edit window derived from a WT/edited sequence pair.

    ``ref`` and ``alt`` are the segments that remain after stripping the shared
    prefix and suffix, so ``wt == prefix + ref + suffix`` and
    ``edited == prefix + alt + suffix``.
    """

    start: int
    ref: str
    alt: str
    edit_type: str
    edit_length: int
    n_mismatch: int


def diff_window(wt: str, edited: str) -> EditSpec:
    """Locate the minimal changed window between an unedited and edited sequence."""
    wt, edited = wt.upper(), edited.upper()
    n = min(len(wt), len(edited))

    pre = 0
    while pre < n and wt[pre] == edited[pre]:
        pre += 1

    suf = 0
    while suf < n - pre and wt[len(wt) - 1 - suf] == edited[len(edited) - 1 - suf]:
        suf += 1

    ref = wt[pre : len(wt) - suf]
    alt = edited[pre : len(edited) - suf]

    if not ref and not alt:
        return EditSpec(pre, "", "", "none", 0, 0)
    if len(ref) == len(alt):
        n_mm = sum(a != b for a, b in zip(ref, alt))
        edit_type = "substitution" if n_mm == 1 else "multi_substitution"
        return EditSpec(pre, ref, alt, edit_type, len(ref), n_mm)
    if not ref:
        return EditSpec(pre, ref, alt, "insertion", len(alt), 0)
    if not alt:
        return EditSpec(pre, ref, alt, "deletion", len(ref), 0)
    return EditSpec(pre, ref, alt, "complex", max(len(ref), len(alt)), 0)


def align_pair(wt: str, edited: str) -> tuple[str, str]:
    """Return gapped, equal-length versions of the WT and edited sequences.

    Positions outside the changed window align one-to-one. Inside the window,
    equal-length segments align one-to-one (which is correct for contiguous and
    non-contiguous substitutions); otherwise the shorter segment is right-padded
    with gaps, which encodes insertions and deletions explicitly.
    """
    spec = diff_window(wt, edited)
    pre = wt[: spec.start]
    suf_len = len(wt) - spec.start - len(spec.ref)
    suf = wt[len(wt) - suf_len :] if suf_len else ""

    width = max(len(spec.ref), len(spec.alt))
    ref = spec.ref.ljust(width, GAP)
    alt = spec.alt.ljust(width, GAP)
    return pre + ref + suf, pre + alt + suf


def find_protospacer(target: str, spacer: str) -> tuple[int, str] | None:
    """Locate the genomic protospacer for a designed spacer inside a target window.

    Library spacers are sometimes prefixed with a non-genomic 5' G to satisfy the U6
    promoter, so both the raw spacer and the G-trimmed form are tried. The leftmost
    match wins: some target windows contain a tandem duplication of the site, where
    preferring the longer candidate would select the downstream copy and place the
    edit tens of bases from the nick.
    """
    target, spacer = target.upper(), spacer.upper()
    candidates = [spacer]
    if spacer.startswith("G"):
        candidates.append(spacer[1:])

    best: tuple[int, str] | None = None
    for candidate in candidates:
        idx = target.find(candidate)
        if idx < 0:
            continue
        if best is None or idx < best[0] or (idx == best[0] and len(candidate) > len(best[1])):
            best = (idx, candidate)
    return best
