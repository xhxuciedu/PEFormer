# E01 - canonical decision manifest and candidate availability

Descriptive audit of the frozen corpus. Nothing trained, tuned or selected.

## Canonicalisation

| quantity | count |
|---|---:|
| rows | 318,471 |
| distinct raw (WT, edited) string pairs | 139,503 |
| stored target-site names | 42,774 |
| distinct protospacers | 39,040 |
| **canonical intended alleles** | 51,766 |
| distinct pegRNA design molecules | 145,934 |
| distinct experimental contexts | 28 |
| **decision groups (allele x context)** | 175,668 |
| rows with a full 12 bp flank on both sides | 89.1% |
| target sites whose several WT windows are one allele | 32,181 |
| alleles reachable by more than one protospacer | 8,024 |

The raw string pair is a design artefact: collapsing the window convention takes 130,921 distinct WT windows down to 51,766 intended alleles, and merges designs that reach the same allele from either strand.

## Candidate depth per decision group

| surface | rows | decision groups | >=2 designs | >=3 | >=5 | rows in >=2 |
|---|---:|---:|---:|---:|---:|---:|
| corpus | 318,471 | 175,668 | 54,390 | 18,332 | 9,543 | 196,853 |
| fold0_heldout | 20,509 | 17,872 | 2,412 | 85 | 0 | 5,013 |
| fold1_val | 59,581 | 35,457 | 11,559 | 4,633 | 912 | 35,647 |
| folds2_5_train | 238,381 | 132,066 | 42,238 | 14,633 | 7,149 | 148,416 |

**The held-out surface cannot support the manuscript's decision metric at its own candidate depth.** Fold 0 contains 2,412 groups with two or more alternative designs for one allele in one context, 85 with three or more, and **0 with five or more**. The manuscript's 735 protospacer groups of >=5 rows are a different object: 100.0% of them span more than one source/cell/editor condition and 25.6% contain more than one intended allele (median 1 alleles per group).

## True replicates

- metadata-identical replicate groups: **654** covering 1,512 rows
- size profile: {2: 586, 5: 68}
- groups spanning two source files: 196
- median within-group SD of measured efficiency: 0.0032 (mean of group means 0.1434)

A replicate here is metadata-identical: same canonical allele, same design molecule, same nine context fields. Two rows can still be the same library measured twice or two genuinely independent transfections; the corpus does not record which, so these are an upper bound on identified replication and a lower bound on measurement noise.

## Fold structure under the canonical keys

| key | keys touching fold 0 | also present in other folds | share |
|---|---:|---:|---:|
| target_name | 808 | 157 | 19.4% |
| spacer | 750 | 605 | 80.7% |
| edit_key | 3,115 | 339 | 10.9% |
| decision_group | 17,872 | 3,670 | 20.5% |

4,931 of 20,509 fold-0 rows (24.0%) sit in a decision group that the training folds also contain.

OptiPrime's official folds are not locus-disjoint under the canonical keys: 80.7% of fold-0 protospacers and 20.3% of fold-0 decision groups also appear in training folds. The manuscript's leakage check counted exact design-and-condition twins (196 rows); alternative designs for the same intended allele at the same site are a separate and much larger channel.

## Consequence for the plan

1. A fixed-allele selection benchmark at >=5 candidates does not exist on fold 0. It exists on the development folds (fold 1 alone: 912 groups) and in the corpus as a whole (9,543 groups).

2. On fold 0 the identifiable decision is the **pairwise** one: two alternative designs for the same allele in the same context. That is a real estimand and E02 reports it.

3. Locus-disjoint splits must be rebuilt on the canonical edit key, not the protospacer and not `target_name`, before any transfer claim.
