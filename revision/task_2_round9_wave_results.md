# Round-9 experimental waves

Matched development folds, controls run in the same wave at the identical recipe and
seed. Commit `5cc6fe93a649`.

## C2 — censoring-aware ordinal loss: falsified, and the control beat it

Control `r9_ctrl` = 0.8978 (dev fold 0). Thresholds on the dev folds
number 18; the table gives how many are masked for a row measured at exactly zero.

| variant | limit | masked / 18 | best val ρ | Δ vs control |
|---|---:|---:|---:|---:|
| shuffle control (3 random terms) | — | 3 | 0.8990 | +0.0012 |
| censor below p90 | 0.00159 | 2 | 0.8946 | -0.0032 |
| censor below p95 | 0.00307 | 3 | 0.8899 | -0.0079 |
| censor below p99 | 0.01095 | 4 | 0.8897 | -0.0081 |

Two things kill this mechanism.

**A dose-response in the wrong direction.** Masking more of the low tail monotonically
hurts: −0.0032 at two thresholds, −0.0079 at three, −0.0081 at four. If censored zeros
were noise, removing more of them should have helped.

**The mechanism-free control beats every real variant.** The shuffle control drops the
same *number* of terms per zero row, chosen at random, and scores
0.8990 — above the plain control and
0.0091 above the matched p95 variant
that drops the same count structurally. So dropping terms at random is harmless-to-mildly
helpful, and dropping *specifically the lowest thresholds* is what costs performance.

The interpretation is that the zero-versus-just-above-zero distinction carries real
ranking signal, and the censoring correction throws it away. The replicate evidence that
21.4% of measured zeros have a non-zero replicate is still true; it just does not follow
that the model should stop learning that boundary. This is the third time in this project
a mechanism-free control has matched or beaten the mechanism it was built to isolate.

## W1 — weighting replication across three folds

Positive Δ means uniform weighting beats OptiPrime's row weights.

| dev fold | control (uniform) | OptiPrime weights | Δρ |
|---:|---:|---:|---:|
| 0 | 0.8978 | 0.8830 | +0.0148 |
| 1 | pending | pending | pending |
| 2 | pending | pending | pending |

Mean over 1 fold(s): **+0.0148**, range [+0.0148, +0.0148], same sign on every fold.
