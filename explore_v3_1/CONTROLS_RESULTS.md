# V3.1 controls: completed results

One-subset, one-seed retrospective development; unadjusted paired component intervals.

All configurations and checkpoints are selected using inner data only. Source-constrained
selection uses the two prespecified source-validation studies, not source fold 0.
Each fit has 1001 total labelled target groups, including 162 inner-validation groups.

Released OptiPrime target @1: 0.041409; original source @1: 0.133635.

| Method | Policy | Horizon / LR multiplier | Selected step | Inner @1 | Target @1 | Delta vs OptiPrime [95% CI] | Source audit @1 |
|---|---|---|---:|---:|---:|---|---:|
| A | target | 100 / 3 | 70 | 0.040936 | 0.041101 | -0.000309 [-0.000922, +0.000271] | 0.128339 |
| A | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| B | target | 500 / 3 | 300 | 0.040682 | 0.040664 | -0.000745 [-0.001438, -0.000104] | 0.119120 |
| B | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| D | target | 500 / 1 | 350 | 0.041177 | 0.039265 | -0.002144 [-0.002859, -0.001408] | 0.111535 |
| D | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| C_frozen | target | 500 / 1 | 400 | 0.042260 | 0.038883 | -0.002526 [-0.003212, -0.001794] | 0.122010 |
| C_frozen | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| E | target | 100 / 1 | 60 | 0.041426 | 0.041304 | -0.000105 [-0.000697, +0.000487] | 0.133085 |
| E | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |

## Direct contrasts

| Policy | Contrast | Surface | Difference | 95% CI |
|---|---|---|---:|---|
| target | E minus A | target_outer_val | +0.000203 | [-0.000295, +0.000736] |
| target | E minus A | source_val | -0.004050 | [-0.006810, -0.001497] |
| target | E minus A | source_audit | +0.004746 | [+0.002327, +0.007184] |
| target | E minus C_frozen | target_outer_val | +0.002421 | [+0.001776, +0.003054] |
| target | E minus C_frozen | source_val | +0.017056 | [+0.011417, +0.022877] |
| target | E minus C_frozen | source_audit | +0.011075 | [+0.007924, +0.014395] |
| constrained | E minus A | target_outer_val | +0.000000 | [+0.000000, +0.000000] |
| constrained | E minus A | source_val | +0.000000 | [+0.000000, +0.000000] |
| constrained | E minus A | source_audit | +0.000000 | [+0.000000, +0.000000] |
| constrained | E minus B | target_outer_val | +0.000000 | [+0.000000, +0.000000] |
| constrained | E minus B | source_val | +0.000000 | [+0.000000, +0.000000] |
| constrained | E minus B | source_audit | +0.000000 | [+0.000000, +0.000000] |

## All fit configurations and endpoint trajectories

| Arm / source loss / sampling / weight | Horizon | LR | Target-selected step | Target-selected outer @1 | Final inner @1 | Final outer @1 | Target passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| A / none / mixture / 1 | 100 | 1 | 10 | 0.039302 | 0.038582 | 0.040528 | 7.70 |
| A / none / mixture / 1 | 100 | 3 | 70 | 0.041101 | 0.040477 | 0.041368 | 7.70 |
| A / none / mixture / 1 | 500 | 1 | 150 | 0.040830 | 0.038065 | 0.040337 | 38.48 |
| A / none / mixture / 1 | 500 | 3 | 100 | 0.041368 | 0.038968 | 0.040581 | 38.48 |
| B / none / mixture / 1 | 100 | 1 | 10 | 0.039913 | 0.037862 | 0.040178 | 7.70 |
| B / none / mixture / 1 | 100 | 3 | 70 | 0.041358 | 0.037658 | 0.041152 | 7.70 |
| B / none / mixture / 1 | 500 | 1 | 250 | 0.040336 | 0.039371 | 0.039808 | 38.48 |
| B / none / mixture / 1 | 500 | 3 | 300 | 0.040664 | 0.040602 | 0.040289 | 38.48 |
| C_frozen / none / mixture / 1 | 100 | 1 | 20 | 0.041515 | 0.039896 | 0.040798 | 7.70 |
| C_frozen / none / mixture / 1 | 100 | 3 | 20 | 0.041411 | 0.039617 | 0.039996 | 7.70 |
| C_frozen / none / mixture / 1 | 500 | 1 | 400 | 0.038883 | 0.040271 | 0.038747 | 38.48 |
| C_frozen / none / mixture / 1 | 500 | 3 | 200 | 0.039608 | 0.037379 | 0.038999 | 38.48 |
| D / none / mixture / 1 | 100 | 1 | 60 | 0.041665 | 0.040353 | 0.040529 | 7.70 |
| D / none / mixture / 1 | 100 | 3 | 40 | 0.041465 | 0.038552 | 0.040365 | 7.70 |
| D / none / mixture / 1 | 500 | 1 | 350 | 0.039265 | 0.040613 | 0.039463 | 38.48 |
| D / none / mixture / 1 | 500 | 3 | 50 | 0.041977 | 0.040203 | 0.040120 | 38.48 |
| E / none / mixture / 1 | 100 | 1 | 60 | 0.041304 | 0.037839 | 0.040718 | 7.70 |
| E / none / mixture / 1 | 100 | 3 | 50 | 0.041197 | 0.039407 | 0.040132 | 7.70 |
| E / none / mixture / 1 | 500 | 1 | 450 | 0.038688 | 0.040650 | 0.038712 | 38.48 |
| E / none / mixture / 1 | 500 | 3 | 150 | 0.039504 | 0.039889 | 0.039096 | 38.48 |

20 fits; 0.781 summed process-hours including evaluation/I/O.
The point-estimate source feasibility filter is not formal noninferiority.
These are single-model outcomes, not seed means or score ensembles. No independent confirmation is implied.
