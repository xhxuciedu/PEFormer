# Task 1.2 — deployment utility per target

735 targets with >= 5 designs, covering
20,474 of the 20,509 held-out rows. Bootstrap resamples targets
(= protospacer clusters), 5000 resamples, seed 20260903. Commit
`d47d183082cb`.

Random selection is the expectation over designs at that target, computed exactly rather
than simulated, except NDCG@5 which is averaged over 40 random permutations.

| metric | random | OptiPrime | PE-RankFormer | Δ | 95% CI | p | targets |
|---|---:|---:|---:|---:|---|---:|---:|
| precision@1 | 0.051 | 0.224 | **0.367** | +0.143 | [+0.106, +0.180] | 0.0002 | 735 |
| precision@5 | 0.255 | 0.504 | **0.573** | +0.069 | [+0.055, +0.082] | 0.0002 | 735 |
| NDCG@5 | 0.300 | 0.677 | **0.765** | +0.088 | [+0.071, +0.104] | 0.0002 | 670 |
| efficiency of the top-1 pick | 0.082 | 0.220 | **0.235** | +0.015 | [+0.009, +0.022] | 0.0002 | 735 |
| regret vs best design | 0.187 | 0.049 | **0.033** | -0.015 | [-0.022, -0.009] | 0.0002 | 735 |
| top-1 achieves >= 5% | 0.290 | 0.529 | **0.540** | +0.011 | [-0.003, +0.026] | 0.1536 | 735 |
| top-1 achieves >= 20% | 0.142 | 0.370 | **0.396** | +0.026 | [+0.010, +0.044] | 0.0024 | 735 |

Regret is in units of editing efficiency: how much efficiency is lost by taking the
model's first pick instead of the target's best design. Lower is better, so a negative Δ
favours PE-RankFormer for that row.

Per-target table: `task_1_2_utility_table.csv`.
