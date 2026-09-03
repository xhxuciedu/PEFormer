# Task 1.1 — per-target and per-condition Spearman

Frozen-prediction re-analysis. Protospacer-clustered bootstrap, 5000 resamples,
seed 20260903. Commit `d47d183082cb`.

## The headline number falls a long way when the locus effect is removed

| evaluation | OptiPrime | PE-RankFormer | Δρ | 95% CI |
|---|---:|---:|---:|---|
| Pooled over all 20,509 rows | 0.8690 | 0.9079 | +0.0389 | see task 1.7 / Table 2 |
| Within condition, n-weighted (14 conditions) | 0.7605 | 0.8111 | +0.0506 | [+0.0345, +0.0670] |
| **Within target**, mean over 670 targets | **0.5472** | **0.6356** | **+0.0884** | [+0.0761, +0.1007] |
| Within target, n-weighted | — | — | +0.0758 | [+0.0649, +0.0874] |

Targets scored: 670 of 750 protospacers had
>= 5 designs and a non-constant target, covering 19,334 rows
(median 18 designs per target,
range 6-200).

## Distribution of within-target Spearman

| | p10 | q1 | median | q3 | p90 | mean |
|---|---:|---:|---:|---:|---:|---:|
| PE-RankFormer | 0.087 | 0.434 | 0.772 | 0.895 | 0.944 | 0.636 |
| OptiPrime | -0.017 | 0.348 | 0.670 | 0.811 | 0.873 | 0.547 |
| difference | -0.056 | +0.010 | +0.082 | +0.161 | +0.258 | +0.088 |

PE-RankFormer is ahead on 78.4% of scored
targets, and ahead in 100.0% of bootstrap resamples
(p = 0.0002).

Per-target table: `task_1_1_per_target_table.csv`.
