"""Canonical intended-edit keys from the frozen corpus windows.

The corpus stores a variable-length local window (`full_unedited` / `full_edited`) whose
extent depends on PBS/RTT length, so the exact string pair is a *design* artefact, not an
edit identity: 19,282 of 42,774 target sites carry four different WT strings. Anything
that groups candidate pegRNAs by the raw pair therefore oversplits equivalent edits.

This module reduces each row to (ref allele, alt allele) plus a fixed flank of WT context,
left-normalised for indels and strand-normalised, which is the closest available stand-in
for "genomic coordinate + REF/ALT" when no coordinates are stored.
"""
from __future__ import annotations

import pandas as pd

# Bump whenever the key definition changes. `_v2common.load_corpus` puts this in the
# cache filename, so a stale manifest cannot be silently reused by newer code.
CANON_VERSION = 2

COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def minimal_edit(wt: str, ed: str) -> tuple[int, str, str]:
    """Strip the shared prefix and suffix. Returns (offset in wt, ref, alt)."""
    i = 0
    n = min(len(wt), len(ed))
    while i < n and wt[i] == ed[i]:
        i += 1
    j = 0
    while j < n - i and wt[len(wt) - 1 - j] == ed[len(ed) - 1 - j]:
        j += 1
    return i, wt[i:len(wt) - j], ed[i:len(ed) - j]


def left_normalise(wt: str, pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """VCF-style left shift for indel representations.

    A pure insertion or deletion in a repeat can be written at several offsets; shifting
    as far left as the flanking base allows makes those representations identical. Only
    applied when one allele is empty, which is where the ambiguity lives.
    """
    if ref and alt:
        return pos, ref, alt
    while pos > 0:
        base = wt[pos - 1]
        seg = ref or alt
        if seg[-1] != base:
            break
        seg = base + seg[:-1]
        pos -= 1
        if ref:
            ref = seg
        else:
            alt = seg
    return pos, ref, alt


def edit_type(ref: str, alt: str) -> str:
    if not ref and not alt:
        return "none"
    if len(ref) == len(alt) == 1:
        return "sub1"
    if len(ref) == len(alt):
        return f"sub{len(ref)}"
    if not ref:
        return f"ins{len(alt)}"
    if not alt:
        return f"del{len(ref)}"
    return f"complex{len(ref)}_{len(alt)}"


def _oriented_key(wt: str, ed: str, flank: int, ext: str | None = None) -> tuple:
    """One orientation's key: minimal edit, left-normalised *in this orientation*.

    Version 1 of this module normalised the indel on the forward strand only and then
    reverse-complemented that single representation. An indel inside a repeat has its own
    leftmost position in each orientation, so the two keys disagreed: for
    WT=GGGACACACTTT -> GGGACACTTT the forward and reverse-complement inputs returned
    different keys, and 61.7% of the corpus's 51,012 distinct indel sequence pairs were
    affected. Substitutions were always fine. Each orientation is now normalised
    independently and the minimum taken, which is invariant by construction.
    """
    pos, ref, alt = left_normalise(wt, *minimal_edit(wt, ed))
    src = wt if ext is None else ext
    off = 0 if ext is None else (ext.find(wt) if wt in ext else 0)
    p = pos + off
    left = src[max(0, p - flank):p]
    right = src[p + len(ref):p + len(ref) + flank]
    return f"{left}|{ref}>{alt}|{right}", pos, ref, alt, len(left), len(right)


def canonical_keys(wt: str, ed: str, flank: int = 12) -> dict:
    """Canonical edit identity for one row, invariant to which strand it is written on.

    `edit_key` fixes the allele change and `flank` bp of WT on each side. Both orientations
    are normalised independently and the lexicographic minimum is taken, so two pegRNAs
    approaching the same allele from opposite strands land in the same group.
    `flank_avail` records how much context the window actually supplied; keys built from
    a truncated flank are weaker evidence of identity and are reported separately.
    """
    fwd, pos, ref, alt, nl, nr = _oriented_key(wt, ed, flank)
    rev = _oriented_key(revcomp(wt), revcomp(ed), flank)[0]
    return {
        "edit_pos_in_window": pos,
        "ref": ref,
        "alt": alt,
        "edit_type": edit_type(ref, alt),
        "edit_key": min(fwd, rev),
        "flank_left_avail": nl,
        "flank_right_avail": nr,
        "flank_full": int(nl == flank and nr == flank),
    }


def add_canonical(df: pd.DataFrame, flank: int = 12) -> pd.DataFrame:
    """Vectorised-enough wrapper: one pass over the unique (wt, ed) pairs.

    The corpus has 318,471 rows but far fewer distinct sequence pairs, and the string
    work is pure, so caching on the pair keeps this a few seconds rather than minutes.
    """
    pairs = pd.unique(df.full_unedited + "\x00" + df.full_edited)
    recs = {}
    for p in pairs:
        wt, ed = p.split("\x00")
        recs[p] = canonical_keys(wt, ed, flank)
    key = df.full_unedited + "\x00" + df.full_edited
    out = pd.DataFrame([recs[k] for k in key], index=df.index)
    return pd.concat([df, out], axis=1)


# --------------------------------------------------------------------------- #
# design / context / replicate keys
# --------------------------------------------------------------------------- #
# The design/context boundary follows round 6: a field is DESIGN if changing it changes
# the molecule the user orders, and CONTEXT if it changes the machinery or the biology the
# user is deploying into. `linker` is part of the ordered epegRNA, so it is design.
DESIGN_FIELDS = ["spacer", "pbs", "rtt", "motif", "scaffold_name", "linker", "epegRNA"]
CONTEXT_FIELDS = ["source_study", "cell_type", "pe_type", "cas9_type", "cas9_pam",
                  "PEmax", "MLH1dn", "NRCH", "time"]


def _join(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """String-join a set of columns, with missing values kept as an explicit token.

    `linker` is null for the plain-pegRNA rows; letting `astype(str)` turn that into the
    float `nan` would silently break the join, and dropping it would merge constructs that
    differ.
    """
    out = df[fields[0]].astype("string").fillna("NA")
    for f in fields[1:]:
        out = out + "|" + df[f].astype("string").fillna("NA")
    return out.astype(str)


def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["design_key"] = _join(df, DESIGN_FIELDS)
    df["context_key"] = _join(df, CONTEXT_FIELDS)
    df["cond3"] = _join(df, ["source_study", "cell_type", "pe_type"])
    # A decision group is "one intended allele in one experimental context": the unit a
    # user actually faces. Candidates within it are the alternative pegRNA designs.
    df["decision_group"] = df.edit_key + "@@" + df.context_key
    # True replicates: the same molecule, the same context, the same intended allele.
    df["replicate_key"] = df.decision_group + "##" + df.design_key
    return df


# --------------------------------------------------------------------------- #
# window extension
# --------------------------------------------------------------------------- #
def extended_windows(df: pd.DataFrame) -> pd.Series:
    """Map each row's WT window onto the longest window that extends it.

    Within a target site the stored windows are nested prefixes -- they all start at the
    same upstream anchor and end wherever that row's RTT ends -- so a row whose edit sits
    near its own window's 3' end can borrow downstream context from a longer-RTT row at
    the same site. 42,285 of 42,774 sites are perfectly nested; the remaining 489 are
    handled by the same rule without a special case, since a window that is not a prefix
    of the longest one simply extends only to the longest window it *is* a prefix of.

    This is bookkeeping, not imputation: every extra base is observed WT sequence from the
    same site.
    """
    ext = {}
    for _, s in df.groupby("target_name", observed=True).full_unedited:
        ws = sorted(set(s), key=len)
        for i, w in enumerate(ws):
            best = w
            for w2 in ws[i + 1:]:
                if w2.startswith(w) and len(w2) > len(best):
                    best = w2
            ext[(_, w)] = best
    return pd.Series([ext[(t, w)] for t, w in zip(df.target_name, df.full_unedited)],
                     index=df.index)


def add_canonical_extended(df: pd.DataFrame, flank: int = 12) -> pd.DataFrame:
    """`add_canonical`, but with flanks read off the site's longest observed window.

    Without this, two rows implementing the identical allele at the identical site get
    different keys purely because one row's RTT stopped sooner, which is a design choice.
    """
    wt_ext = extended_windows(df)
    base = add_canonical(df, flank)
    recs = {}
    pairs = pd.unique(wt_ext + "\x00" + df.full_unedited + "\x00" + df.full_edited)
    for p in pairs:
        we, wt, ed = p.split("\x00")
        fwd, _, _, _, nl, nr = _oriented_key(wt, ed, flank, we)
        rev = _oriented_key(revcomp(wt), revcomp(ed), flank, revcomp(we))[0]
        recs[p] = (min(fwd, rev), nl, nr)
    key = wt_ext + "\x00" + df.full_unedited + "\x00" + df.full_edited
    vals = [recs[k] for k in key]
    base["edit_key"] = [v[0] for v in vals]
    base["flank_left_avail"] = [v[1] for v in vals]
    base["flank_right_avail"] = [v[2] for v in vals]
    base["flank_full"] = [int(v[1] == flank and v[2] == flank) for v in vals]
    return base
