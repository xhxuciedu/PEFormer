# E30: pairwise-loss replication

E25 validation; no test results read

Selected-epoch validation values are descriptive, not independent confirmation. Three-seed averages are not ensembles.

Pending checkpoints: none

| Model | Validation @1 | Best epoch (zero-indexed) | Epochs run | Training seconds |
|---|---:|---:|---:|---:|
| P_s20260910 | 0.04269 | 10 | 12 | 995.3 |
| S_s20260910 | 0.04274 | 11 | 12 | 1189.9 |
| shared_s20260910 | 0.04310 | 3 | 8 | 948.7 |
| S_pairwise_s20260910 | 0.04300 | 4 | 9 | 1627.6 |
| P_s20260911 | 0.04273 | 10 | 12 | 975.9 |
| S_s20260911 | 0.04256 | 11 | 12 | 1454.2 |
| shared_s20260911 | 0.04309 | 3 | 8 | 1074.1 |
| S_pairwise_s20260911 | 0.04299 | 5 | 10 | 624.4 |
| P_s20260912 | 0.04272 | 10 | 12 | 991.7 |
| S_s20260912 | 0.04271 | 7 | 12 | 1230.1 |
| shared_s20260912 | 0.04293 | 4 | 9 | 981.3 |
| S_pairwise_s20260912 | 0.04301 | 4 | 9 | 777.9 |

| Contrast | Seeds | Per-seed differences | Mean difference |
|---|---|---|---:|
| pairwise_minus_P | [20260910, 20260911, 20260912] | +0.00031, +0.00026, +0.00029 | +0.00029 |
| pairwise_minus_S | [20260910, 20260911, 20260912] | +0.00026, +0.00042, +0.00030 | +0.00033 |
| pairwise_minus_shared | [20260910, 20260911, 20260912] | -0.00010, -0.00010, +0.00008 | -0.00004 |