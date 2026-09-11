# V3 matched controls and uniform ensembles

Exploratory post-replication checks. Ensembles use all three seeds with equal score weights, not OptiPrime. No weight tuning.

## Matched frozen-backbone controls

| Budget | Control | Reference | Target @1 | Target difference [95% CI] | Source @1 | Source difference [95% CI] |
|---:|---|---|---:|---|---:|---|
| 200 | C_frozen | E | 0.038378 | -0.000302 [-0.000669, +0.000030] | 0.123504 | -0.006283 [-0.008979, -0.003712] |
| 200 | F_replay | F | 0.040287 | +0.000236 [-0.000225, +0.000676] | 0.128018 | +0.003487 [-0.000274, +0.007418] |
| 1000 | C_frozen | E | 0.041515 | +0.000211 [-0.000104, +0.000537] | 0.127078 | -0.006008 [-0.008830, -0.003311] |
| 1000 | F_replay | F | 0.040866 | -0.000291 [-0.000748, +0.000143] | 0.128611 | -0.001381 [-0.004820, +0.002285] |

Controls have one seed. C_frozen removes encoder-update/capacity differences from E,
but not initialization or initial score-scale differences. Source-label replay and soft-teacher
KL have different gradient scales; equal weights do not establish mechanistic equivalence.

## Equal-weight three-seed score ensembles

| Budget | Arm | Target @1 | Difference vs OptiPrime [95% CI] | Source @1 | Source change [95% CI] |
|---:|---|---:|---|---:|---|
| 200 | A | 0.039390 | -0.002020 [-0.002655, -0.001370] | 0.131755 | -0.001880 [-0.004100, +0.000256] |
| 200 | D | 0.039169 | -0.002240 [-0.002924, -0.001541] | 0.115537 | -0.018097 [-0.022144, -0.014115] |
| 200 | E | 0.038830 | -0.002579 [-0.003278, -0.001888] | 0.128293 | -0.005342 [-0.008178, -0.002610] |
| 200 | F | 0.040265 | -0.001144 [-0.001784, -0.000484] | 0.124428 | -0.009207 [-0.012259, -0.006338] |
| 1000 | A | 0.039720 | -0.001689 [-0.002318, -0.001090] | 0.129600 | -0.004034 [-0.006777, -0.001406] |
| 1000 | D | 0.041665 | +0.000256 [-0.000312, +0.000873] | 0.112177 | -0.021458 [-0.026209, -0.017063] |
| 1000 | E | 0.041312 | -0.000097 [-0.000696, +0.000489] | 0.133306 | -0.000329 [-0.002496, +0.001868] |
| 1000 | F | 0.041147 | -0.000262 [-0.000852, +0.000296] | 0.129522 | -0.004112 [-0.007156, -0.001197] |

These are actual score ensembles, distinct from mean seed outcomes. A/D need three
adapted backbones; E/F can share one frozen backbone with three scalar heads. No fitted
ensemble weights and no additional target labels. Intervals are exploratory, unadjusted
component bootstrap intervals on the same exposed development evaluation.
