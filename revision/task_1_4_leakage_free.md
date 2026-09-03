# Task 1.4 — leakage-free subset

A held-out row is "twinned" if some training row matches it on all sixteen design and
condition covariates **and** the target site — the same key the replicate analysis uses.

196 of 20,509 held-out rows
(1.0%) have such a twin in training. Removing them leaves
20,313 rows over 741 protospacers.

| subset | rows | protospacers | OptiPrime | PE-RankFormer | Δρ | 95% CI on Δ | 95% CI on ours |
|---|---:|---:|---:|---:|---:|---|---|
| Full held-out set | 20,509 | 750 | 0.8690 | 0.9079 | +0.0389 | [+0.0288, +0.0498] | [0.8959, 0.9174] |
| **Leakage-free subset** | 20,313 | 741 | 0.8690 | 0.9075 | +0.0385 | [+0.0286, +0.0494] | [0.8952, 0.9170] |
| Twinned rows only | 196 | 11 | 0.8445 | 0.8970 | +0.0525 | [-0.0360, +0.1845] | [0.6159, 0.9687] |

Protospacer-clustered bootstrap, 5000 resamples, seed 20260903.
Commit `d47d183082cb`.
