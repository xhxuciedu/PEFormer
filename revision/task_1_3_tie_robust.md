# Task 1.3 — tie-robust and decomposed metrics

Protospacer-clustered bootstrap, 2000 resamples, seed 20260903.
Commit `d47d183082cb`.

## Tie structure: is the comparison like-for-like?

| vector | distinct values over 20,509 rows | largest tie block |
|---|---:|---:|
| PE-RankFormer | 19,794 | 0.03% |
| OptiPrime | 20,437 | 0.02% |
| measured target | 14,887 | 26.82% |

Both predictors are effectively tie-free, so the Spearman comparison is like-for-like on
the manuscript's own criterion, and Kendall τ-b should agree with it in direction.

## Metrics by partition

| partition | metric | OptiPrime | PE-RankFormer | Δ | 95% CI |
|---|---|---:|---:|---:|---|
| all (n=20,509) | Spearman ρ | 0.8690 | 0.9079 | +0.0389 | — |
| all (n=20,509) | Kendall τ-b | 0.6931 | 0.7478 | +0.0547 | [+0.0426, +0.0674] |
| all (n=20,509) | AUROC (edits at all) | 0.9173 | 0.9465 | +0.0292 | [+0.0220, +0.0367] |
| all (n=20,509) | Spearman ρ | y>0 | 0.8256 | 0.8725 | +0.0469 | [+0.0340, +0.0613] |
| Liu (n=9,175) | Spearman ρ | 0.8365 | 0.8585 | +0.0220 | — |
| Liu (n=9,175) | Kendall τ-b | 0.6497 | 0.6741 | +0.0244 | [+0.0015, +0.0489] |
| Liu (n=9,175) | AUROC (edits at all) | 0.8175 | 0.8410 | +0.0236 | [-0.0109, +0.0496] |
| Liu (n=9,175) | Spearman ρ | y>0 | 0.8365 | 0.8582 | +0.0217 | [+0.0021, +0.0419] |
| Kim (n=11,334) | Spearman ρ | 0.7320 | 0.8124 | +0.0803 | — |
| Kim (n=11,334) | Kendall τ-b | 0.5717 | 0.6620 | +0.0903 | [+0.0751, +0.1067] |
| Kim (n=11,334) | AUROC (edits at all) | 0.8457 | 0.8974 | +0.0516 | [+0.0411, +0.0632] |
| Kim (n=11,334) | Spearman ρ | y>0 | 0.7744 | 0.8618 | +0.0874 | [+0.0654, +0.1082] |

`Spearman ρ | y>0` is restricted to rows that actually edit, so it isolates
quantification from detection; `AUROC (edits at all)` isolates detection.
