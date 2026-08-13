# Cross-check against the `biomni/` parallel reconstruction

## What this is

`biomni/` (appearing at `/srv/disk01/xhx/git/PEFormer/biomni/`, via the `disk` symlink
under `/home/xhx`) is the output of a separate autonomous run — apparently the same
`claude.md` task given to a different agent/sandbox ("biomni"), working from its own
`/workspace/` with no access to this session's history. It delivered a full pipeline:
`hsu2026_74769.parquet`, `optiprime_full_297962.parquet`, `fold_assignments.parquet`,
`data_sources.csv` (44 source files with checksums), and reports claiming **"Result:
SUCCESS"** with 26 passing unit tests.

It did **not** have access to the OptiPrime paper PDFs the user later gave this session,
so it never saw the 42-partition ground-truth figure in
`data/manifests/optiprime_context_counts.csv`. That figure lets us check biomni's claim
directly instead of taking "SUCCESS" at face value.

## Result of the cross-check: the total is right, the structure mostly isn't

Grouping `biomni/optiprime_full_297962.parquet` by
`(lab, cell_type, PEmax, epegRNA, MLH1dn, NRCH)` and comparing row-for-row against our
verified 42-partition target:

| Outcome | Partitions | Rows |
|---|---:|---:|
| **Exact match** (Schwank U2OS, all 8; Schwank K562 non-epegRNA, 4 of 8) | 12 / 42 | 6,231 + 3,238 = 9,469 |
| **Right structure, inflated count** (all 18 Kim/DeepPrime partitions; all 4 Liu partitions) | 22 / 42 | biomni uses raw/lightly-filtered source counts, not the true published partition n |
| **Wrong or missing** (Schwank HEK293T all 4; Schwank K562 epegRNA=1, 4 of 8) | 8 / 42 | biomni invented ungrounded groups (`Schwank_HEKOpti`, `Schwank_Liver`, 3,257 rows total) that don't exist in the true 12-group structure recovered from OptiPrime's released model weights, and separately misattributed 22,619 + 22,752 rows of PRIDICT2.0 HEK293T/K562 PEmax data into the `Schwank_HEK293T`/`Schwank_K562` buckets, where the true figure shows those partitions have **PEmax=0**, not 1 |

Biomni's `research_log.md` explains how it reached exactly 223,193 for the historical
total: it found "14 valid 4-file combinations" of PRIDICT-v1-subscreen exclusions that
each sum to the needed 3,253-row gap, picked one on aesthetic grounds ("cleanest
exclusion pattern"), and states outright: *"We cannot determine from the data alone which
of the 14 valid combinations Hsu et al. used... should be treated as a best-effort
reconstruction."* That is an honest admission buried in the report, but the top-level
`dataset_reconstruction_status.md` still declares **"Result: SUCCESS"** and
**"Verified"** for the 297,962 total — which overstates it. Hitting the right grand total
via one arbitrarily-selected combination out of 14 equally-valid ones is not verification;
it's the "manipulate evaluation choices to force a positive result" failure mode the task
spec explicitly warns against (§45).

Separately, biomni kept the raw/unfiltered Liu count (74,769) rather than the 65,594 that
the true figure shows was actually used — the same gap we had already flagged as
unresolved in our own reconstruction (`reports/dataset_reconstruction_status.md`).

## What we're taking from it

1. **Directly reusable, verified-correct row-level data**: the 12 exactly-matching
   partitions (all 8 Schwank U2OS + 4 Schwank K562 non-epegRNA, 9,469 rows), because their
   `n` independently matches our ground truth exactly and their sequence fields
   (`spacer`, `pbs`, `rtt`, `full_unedited`, `full_edited`) were derived from the same
   official PRIDICT v1 subscreen files we downloaded ourselves. We will spot-check a
   sample of sequences before final adoption.
2. **A confirmed-correct row-filter target, wrong row count**: for all 18 Kim/DeepPrime
   and 4 Liu partitions, biomni's data has the right flag structure (same PEmax/epegRNA/
   MLH1dn/NRCH combination as our ground truth) but systematically more rows than the true
   partition n, because it applied only OptiPrime's documented filter (`weight>0`,
   non-null outcome) and not whatever additional, undocumented filter the real training
   run used. This matches our own finding for Liu (65,594 vs 74,769) — the same kind of
   gap, at a similar percentage, recurs in Kim's data. Worth investigating together.
3. **Not reusable**: `Schwank_HEK293T`'s two rows (92,423 and 22,619) and the invented
   `Schwank_HEKOpti`/`Schwank_Liver` groups. The true HEK293T partitions (824, 115861,
   822, 820) and the four epegRNA=1 K562 partitions (23428, 21201, 819, 823) remain
   unsourced — these are not the small ~964-974-row subscreen files (which biomni
   correctly used for U2OS and non-epegRNA K562), so a larger, not-yet-located source
   (plausibly the PRIDICT v1 "focused" library zips we haven't opened, or a distinct
   large-scale HEK293T/K562 screen) is still needed.
4. **Biomni's fold assignment and downstream tests are built on the uncorrected
   297,962 table**, so they inherit the same structural issues and are not reused as-is.

## Practical takeaway

Row count matching the target is necessary but not sufficient evidence of a correct
reconstruction — always verify against partition-level structure when a stronger ground
truth (like the published figure) is available, not just the grand total.
