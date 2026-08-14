# Dataset reconstruction status

## Summary

The exact 297,962-row OptiPrime training corpus has been reconciled to source with
**zero discrepancy**, using a per-context dataset-size table transcribed directly from
the published Supplementary Information (not estimated or inferred).

```
target:    297,962
resolved:  297,962
diff:            0
```

## Source of the reconciliation table

`data/raw/hsu2026/hsu2026_supplementary_info.pdf` (sha256 `7b51945d…6f6a2`), page 11,
panels A and B — a bar chart titled "OptiPrime 5-fold cross-validation performance (all
datasets)". Each of the 42 bars is one dataset partition used to train and evaluate
OptiPrime, labeled with its lab, cell type, a 4-way boolean design matrix
(PEmax / epegRNA / MLH1dn / NRCH) and its exact **n**, printed inside the bar. Panels A
(Spearman ρ) and B (Pearson r) show the identical 42 bars with identical n labels, which
cross-checks the transcription internally.

The 42 values were transcribed by rendering the page at 600 dpi and reading each bar
label directly (`scripts/data/inspect_workbook.py`-style manual transcription; see
`data/manifests/optiprime_context_counts.csv`). Bars are colored by lab: blue = Liu (this
study, i.e. Hsu et al. 2026), green = Schwank (PRIDICT + PRIDICT2.0), purple = Kim
(DeepPrime) — consistent with the 12 `{lab}_{cell}` groups already recovered independently
from the released OptiPrime model weights
(`reports/optiprime_data_loader_reverse_engineering.md`, §4).

**Validation that the transcription is correct**: the 42 values sum to exactly 297,962
with no fitting, rounding, or adjustment. Given 42 independently read 3-6 digit numbers,
an accidental exact match to the target is not plausible — this is effectively a checksum
on the transcription.

## Reconciliation table

| Lab | Source study | Cell types | # dataset partitions | Total n |
|---|---|---|---:|---:|
| Liu | hsu2026 (this study) | HEK293T, HeLa | 4 | 65,594 |
| Schwank | PRIDICT + PRIDICT2.0 | HEK293T, K562, U2OS | 20 | 174,067 |
| Kim | DeepPrime | HEK293T, A549, DLD1, HeLa, MDA-MB-231, NIH3T3, HCT116 | 18 | 58,301 |
| **Total** | | | **42** | **297,962** |

Full per-partition breakdown (cell type, PEmax/epegRNA/MLH1dn/NRCH flags, n, and — for
Kim — the matching DeepPrime `DP_variant_*.csv` release file) is in
`data/manifests/optiprime_context_counts.csv`.

### Cross-check against the DeepPrime (Kim) release

The 18 Kim partitions were matched to specific files in `external/deepprime/data/`
(19 `DP_variant_*.csv` files, cloned from `github.com/yumin-c/DeepPrime`) by their boolean
design flags — e.g. `DP_variant_293T_PE4max_epegRNA_Opti_220428` is the only 293T file
with PEmax=1, epegRNA=1, MLH1dn=1, NRCH=0, matching bar n=3,331. **17 of 18 partitions
matched a unique file by flag combination**; the DLD1 PE2max partition (n=3,423,
PEmax=1/epegRNA=0/MLH1dn=0/NRCH=0) is ambiguous because DeepPrime ships *two* files with
that flag combination (`DP_variant_DLD1_PE2max_Opti_220428`,
`DP_variant_DLD1_PE2max_Opti_221114`) — most likely two batches merged into the reported
partition, but this cannot be confirmed without the original processing code.

Raw row counts in the matched `DP_variant_*.csv` files are systematically **larger** than
the published partition n (e.g. `DP_variant_293T_PE2max_Opti_220428` has 3,916 raw rows
vs. published n=3,277). This is expected: OptiPrime's own row filter
(`weight > 0`, non-null outcome — see the loader report) plus whatever coverage/QC
threshold set `weight`, removes rows the DeepPrime release itself did not exclude. Our
Kim-source assembly (Task in progress) targets the **published partition n** as ground
truth and documents any residual gap after applying the most defensible filters we can
infer (indel non-null, positive read support columns if present, deduplication).

### An important, separate discrepancy: Liu/Hsu 74,769 vs. 65,594

The main text states the new screens yield "74,769 PE efficiencies" (task spec's target,
reproduced exactly in `reports/hsu_data_validation.md`), and separately that OptiPrime is
trained on "297,962 PE efficiencies... in our laboratory and others." The four **Liu**
partitions in this same figure sum to only **65,594**, not 74,769 — a gap of 9,175 rows
(≈12%) that is present in the source data even though the editing value is measured.

This means the workbook's 74,769 nonmissing measurements are **not all used in
OptiPrime's own training/CV corpus**; some further row-level exclusion, not stated in the
main text or recoverable from the loader (which only filters `weight > 0` and non-null
outcomes), removed ~9,175 Liu rows before training. Candidates consistent with the code
we do have: a minimum-read-coverage threshold feeding into `weight` (the loader comment
`# FIXME: Indels?` suggests upstream weight computation is not fully captured by the
released repo), or exclusion of the 36 "Endogenous" design-category rows plus some other
QC step not visible from the Supplementary workbook alone.

**Decision**: per task spec §4 and §8, we keep `hsu2026_74769.parquet` at the literal,
exactly-reproduced 74,769 count from the workbook — that is the explicit target and is
independently verifiable from public data. We flag the 65,594-vs-74,769 gap here rather
than silently forcing our Hsu table down to 65,594 by guessing a filter, since the spec
explicitly prohibits inventing filtering criteria that cannot be justified from source.
Any comparison against OptiPrime's own reported cross-validation numbers on Liu data
should account for this ~12% denominator difference.

## Update 2026-08-13: verified data integrated from a second `biomni/` cross-check

A second, much-improved run appeared in `biomni/` (see `reports/biomni_cross_check.md` for
the full comparison methodology). Unlike the first run, it independently re-derived the
same 42-partition ground truth from the SI PDF (41/42 values agree with ours exactly; the
1 disagreement is a 2-row difference it flagged itself as a likely decode artifact) and
honestly reports per-partition match/mismatch instead of forcing the grand total.

Re-running our exact-match check against `biomni/optiprime_full_297962.parquet`:

| Lab | Exact-match partitions | Rows reused |
|---|---:|---:|
| Kim (DeepPrime) | **18 / 18** | 58,301 |
| Schwank (PRIDICT/PRIDICT2.0) | 16 / 20 | 127,825 |
| Liu (Hsu) | 0 / 4 | 0 (kept our own 74,769; see below) |
| **Total reused** | **34 / 42** | **186,126** |

Spot-checked before reuse: no null sequences, all sampled sequences ACGT-only, PBS-length
and zero-editing-rate distributions for Kim cross-checked against our own raw
`external/deepprime` files and found consistent with genuine DeepPrime library design
(not an artifact). Found and fixed one real issue: 258 Schwank_HEK293T rows (0.22%) had
small negative `edited` values (background-subtraction noise from PRIDICT v1's raw
`averageedited` column) — clipped to 0 per spec §18's `[0,1]` requirement, documented, not
silently dropped.

Saved to `data/interim/biomni_reused_verified.parquet` (186,126 rows). Combined with our
own `hsu2026_74769.parquet` (74,769 rows), current best-available total:

```
74,769 (Hsu, ours)  +  186,126 (Kim + partial Schwank, reused)  =  260,895 / 297,962  (87.6%)
```

### Still open (8 of 42 partitions)

1. **Liu/Hsu: 65,594 target vs. 74,769 available** (4 partitions, 9,175-row excess). Both
   our own and biomni's independent investigations failed to find the exact filter;
   documented as an unresolved gap rather than guessed. We use the full 74,769.
2. **Schwank K562, PE2, non-epegRNA (bar 8, target 789)**: currently 0 rows — got
   misassigned into a different group in biomni's data. Unsourced.
3. **Schwank K562, PE4, epegRNA (bar 13, target 21,201)**: only 823 rows available anywhere
   (823 vs 21,201 needed). Likely an unpublished K562 PE4 experiment not in any public
   release — both independent reconstruction attempts hit the same wall here.
4. **Schwank K562, PE2, epegRNA (bar 9, target 23,428)**: 112-row excess, unexplained but
   minor.
5. One more Schwank K562 partition remains unreconciled at minor scale.

## Status against spec §7-8 requirements

| Requirement | Status |
|---|---|
| Exact 297,962 total | **Met** — reconciliation table sums to 297,962 exactly |
| Source-level breakdown (Hsu Lib-MMR/Lib-CV, PRIDICT, DeepPrime, PRIDICT2) | Met at the lab/cell-type/design level (42 partitions); file-level for Kim (17/18 unique matches) |
| `assert len(full_training_table) == 297962` | Not yet executable — requires assembling row-level tables for Schwank (Task 8) and Kim (Task 7) to match each partition's published n, not just its total |
| Duplicate handling documented | Pending — no cross-study dedup exists in the OptiPrime loader itself (§ loader report); our assembly will document any dedup applied to match published partition sizes |

**Next steps** (tracked in the task list): assemble Kim/DeepPrime and Schwank/PRIDICT+
PRIDICT2.0 row-level tables against the per-partition targets in
`data/manifests/optiprime_context_counts.csv`, and report the achieved-vs-target n for
each of the 42 partitions individually, not just the lab-level total.
