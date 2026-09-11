# V3.1 factorial: completed results

One-subset, one-seed retrospective development; unadjusted paired component intervals.

All configurations and checkpoints are selected using inner data only. Source-constrained
selection uses the two prespecified source-validation studies, not source fold 0.
Each fit has 1001 total labelled target groups, including 162 inner-validation groups.

Released OptiPrime target @1: 0.041409; original source @1: 0.133635.

| Method | Policy | Horizon / LR multiplier | Selected step | Inner @1 | Target @1 | Delta vs OptiPrime [95% CI] | Source audit @1 |
|---|---|---|---:|---:|---:|---|---:|
| kl / balanced / w=0.1 | target | 100 / 1 | 60 | 0.041057 | 0.041399 | -0.000011 [-0.000603, +0.000571] | 0.133425 |
| kl / balanced / w=0.1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| kl / balanced / w=1 | target | 100 / 1 | 100 | 0.041048 | 0.041080 | -0.000330 [-0.000893, +0.000255] | 0.133407 |
| kl / balanced / w=1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| kl / mixture / w=0.1 | target | 100 / 1 | 40 | 0.041126 | 0.040854 | -0.000556 [-0.001164, +0.000029] | 0.132251 |
| kl / mixture / w=0.1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| kl / mixture / w=1 | target | 100 / 1 | 10 | 0.040988 | 0.040894 | -0.000515 [-0.001119, +0.000064] | 0.134306 |
| kl / mixture / w=1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| margin / balanced / w=0.1 | target | 100 / 1 | 40 | 0.040707 | 0.041026 | -0.000383 [-0.000994, +0.000244] | 0.132797 |
| margin / balanced / w=0.1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| margin / balanced / w=1 | target | 100 / 1 | 10 | 0.039939 | 0.040636 | -0.000773 [-0.001410, -0.000141] | 0.134283 |
| margin / balanced / w=1 | constrained | 100 / 1 | 10 | 0.039939 | 0.040636 | -0.000773 [-0.001410, -0.000141] | 0.134283 |
| margin / mixture / w=0.1 | target | 100 / 1 | 30 | 0.040377 | 0.040819 | -0.000590 [-0.001184, -0.000008] | 0.134002 |
| margin / mixture / w=0.1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| margin / mixture / w=1 | target | 100 / 1 | 30 | 0.039723 | 0.040745 | -0.000665 [-0.001251, -0.000049] | 0.132892 |
| margin / mixture / w=1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| utility / balanced / w=0.1 | target | 100 / 1 | 20 | 0.039946 | 0.041231 | -0.000179 [-0.000777, +0.000395] | 0.132029 |
| utility / balanced / w=0.1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| utility / balanced / w=1 | target | 100 / 1 | 40 | 0.040440 | 0.040817 | -0.000592 [-0.001182, +0.000016] | 0.128446 |
| utility / balanced / w=1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| utility / mixture / w=0.1 | target | 100 / 1 | 60 | 0.040568 | 0.041314 | -0.000095 [-0.000665, +0.000489] | 0.129608 |
| utility / mixture / w=0.1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |
| utility / mixture / w=1 | target | 100 / 1 | 40 | 0.040453 | 0.040926 | -0.000483 [-0.001091, +0.000118] | 0.126562 |
| utility / mixture / w=1 | constrained | 100 / 1 | 0 | 0.038033 | 0.039627 | -0.001783 [-0.002435, -0.001158] | 0.133635 |

## Direct contrasts

| Policy | Contrast | Surface | Difference | 95% CI |
|---|---|---|---:|---|
| target | E_h100_lr1_margin_balanced_w1_b1000_sub20260910_s20260910 minus E | target_outer_val | -0.000668 | [-0.001159, -0.000206] |
| target | E_h100_lr1_margin_balanced_w1_b1000_sub20260910_s20260910 minus E | source_val | +0.006237 | [+0.003798, +0.008929] |
| target | E_h100_lr1_margin_balanced_w1_b1000_sub20260910_s20260910 minus E | source_audit | +0.001198 | [-0.000748, +0.003041] |
| constrained | E_h100_lr1_margin_balanced_w1_b1000_sub20260910_s20260910 minus E | target_outer_val | +0.001009 | [+0.000646, +0.001408] |
| constrained | E_h100_lr1_margin_balanced_w1_b1000_sub20260910_s20260910 minus E | source_val | -0.000853 | [-0.001919, +0.000280] |
| constrained | E_h100_lr1_margin_balanced_w1_b1000_sub20260910_s20260910 minus E | source_audit | +0.000648 | [-0.000552, +0.001879] |

## All fit configurations and endpoint trajectories

| Arm / source loss / sampling / weight | Horizon | LR | Target-selected step | Target-selected outer @1 | Final inner @1 | Final outer @1 | Target passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| E / kl / balanced / 0.1 | 100 | 1 | 60 | 0.041399 | 0.039362 | 0.040924 | 7.70 |
| E / kl / balanced / 1 | 100 | 1 | 100 | 0.041080 | 0.041048 | 0.041080 | 7.70 |
| E / kl / mixture / 0.1 | 100 | 1 | 40 | 0.040854 | 0.039775 | 0.040847 | 7.70 |
| E / kl / mixture / 1 | 100 | 1 | 10 | 0.040894 | 0.040486 | 0.041111 | 7.70 |
| E / margin / balanced / 0.1 | 100 | 1 | 40 | 0.041026 | 0.038634 | 0.041111 | 7.70 |
| E / margin / balanced / 1 | 100 | 1 | 10 | 0.040636 | 0.039102 | 0.041344 | 7.70 |
| E / margin / mixture / 0.1 | 100 | 1 | 30 | 0.040819 | 0.038031 | 0.041198 | 7.70 |
| E / margin / mixture / 1 | 100 | 1 | 30 | 0.040745 | 0.039069 | 0.041468 | 7.70 |
| E / utility / balanced / 0.1 | 100 | 1 | 20 | 0.041231 | 0.038952 | 0.041081 | 7.70 |
| E / utility / balanced / 1 | 100 | 1 | 40 | 0.040817 | 0.039961 | 0.040817 | 7.70 |
| E / utility / mixture / 0.1 | 100 | 1 | 60 | 0.041314 | 0.039397 | 0.040922 | 7.70 |
| E / utility / mixture / 1 | 100 | 1 | 40 | 0.040926 | 0.040046 | 0.040808 | 7.70 |

12 fits; 0.175 summed process-hours including evaluation/I/O.
The point-estimate source feasibility filter is not formal noninferiority.
These are single-model outcomes, not seed means or score ensembles. No independent confirmation is implied.
