# V3 initial screen: computed results

All checkpoint choices use budget-internal validation;
the 5,561-group outer validation is an exposed retrospective evaluation, not a new test.
One optimizer seed and one subset seed per arm/budget; intervals are exploratory and unadjusted.

Frozen start target @1: 0.039627; released OptiPrime: 0.041409.
Frozen start source-audit @1: 0.133635 (1463 informative groups).

| Total nominal groups | Arm | Selected step | Target @1 | Delta vs OptiPrime [95% component CI] | Source @1 | Delta vs source start |
|---:|---|---:|---:|---|---:|---:|
| 200 | A: Native ordinal | 10 | 0.039448 | -0.001962 [-0.002597, -0.001311] | 0.131638 | -0.001996 |
| 200 | B: Direct Huber | 80 | 0.038855 | -0.002554 [-0.003237, -0.001856] | 0.116046 | -0.017589 |
| 200 | C: Fresh pairwise | 60 | 0.039337 | -0.002073 [-0.002773, -0.001356] | 0.127399 | -0.006235 |
| 200 | D: Shared Huber + pairwise | 70 | 0.038626 | -0.002783 [-0.003486, -0.002083] | 0.114776 | -0.018859 |
| 200 | E: Anchored pairwise | 80 | 0.038680 | -0.002729 [-0.003419, -0.002016] | 0.129787 | -0.003848 |
| 200 | F: Anchored + preservation | 50 | 0.040052 | -0.001358 [-0.001982, -0.000713] | 0.124532 | -0.009103 |
| 200 | G: Anchored utility + preservation | 10 | 0.039183 | -0.002226 [-0.002857, -0.001585] | 0.132531 | -0.001104 |
| 1000 | A: Native ordinal | 10 | 0.039302 | -0.002108 [-0.002755, -0.001461] | 0.131962 | -0.001673 |
| 1000 | B: Direct Huber | 10 | 0.039913 | -0.001496 [-0.002144, -0.000886] | 0.108704 | -0.024930 |
| 1000 | C: Fresh pairwise | 60 | 0.041334 | -0.000076 [-0.000696, +0.000538] | 0.124735 | -0.008900 |
| 1000 | D: Shared Huber + pairwise | 60 | 0.041665 | +0.000256 [-0.000333, +0.000862] | 0.110664 | -0.022971 |
| 1000 | E: Anchored pairwise | 60 | 0.041304 | -0.000105 [-0.000697, +0.000487] | 0.133085 | -0.000549 |
| 1000 | F: Anchored + preservation | 100 | 0.041157 | -0.000252 [-0.000819, +0.000334] | 0.129991 | -0.003643 |
| 1000 | G: Anchored utility + preservation | 20 | 0.041627 | +0.000217 [-0.000383, +0.000791] | 0.127242 | -0.006392 |

Source @1 is the actual deployed selector, not its original prediction head.
Efficiencies are fractions; multiply differences by 100 for percentage points.

## Direct exploratory contrasts

| Budget | Contrast | Surface | Difference | 95% component interval |
|---:|---|---|---:|---|
| 200 | E minus A | target_outer_val | -0.000767 | [-0.001457, -0.000091] |
| 200 | E minus A | source_val | -0.039024 | [-0.047502, -0.031653] |
| 200 | E minus A | source_audit | -0.001852 | [-0.003717, -0.000051] |
| 200 | F minus A | target_outer_val | +0.000604 | [-0.000052, +0.001278] |
| 200 | F minus A | source_val | -0.012353 | [-0.018675, -0.007175] |
| 200 | F minus A | source_audit | -0.007107 | [-0.010515, -0.003825] |
| 200 | F minus E | target_outer_val | +0.001372 | [+0.000942, +0.001808] |
| 200 | F minus E | source_val | +0.026671 | [+0.018035, +0.036129] |
| 200 | F minus E | source_audit | -0.005255 | [-0.008580, -0.002226] |
| 200 | G minus F | target_outer_val | -0.000868 | [-0.001502, -0.000253] |
| 200 | G minus F | source_val | +0.009161 | [+0.003872, +0.015284] |
| 200 | G minus F | source_audit | +0.007999 | [+0.005106, +0.011165] |
| 200 | D minus C | target_outer_val | -0.000711 | [-0.001357, -0.000072] |
| 200 | D minus C | source_val | -0.014755 | [-0.021345, -0.008726] |
| 200 | D minus C | source_audit | -0.012623 | [-0.016579, -0.008993] |
| 1000 | E minus A | target_outer_val | +0.002003 | [+0.001443, +0.002615] |
| 1000 | E minus A | source_val | -0.007507 | [-0.010442, -0.004959] |
| 1000 | E minus A | source_audit | +0.001123 | [-0.000687, +0.002949] |
| 1000 | F minus A | target_outer_val | +0.001855 | [+0.001267, +0.002476] |
| 1000 | F minus A | source_val | -0.006843 | [-0.012104, -0.002891] |
| 1000 | F minus A | source_audit | -0.001971 | [-0.005202, +0.001162] |
| 1000 | F minus E | target_outer_val | -0.000147 | [-0.000564, +0.000273] |
| 1000 | F minus E | source_val | +0.000663 | [-0.003276, +0.004345] |
| 1000 | F minus E | source_audit | -0.003094 | [-0.006021, -0.000600] |
| 1000 | G minus F | target_outer_val | +0.000470 | [-0.000051, +0.000950] |
| 1000 | G minus F | source_val | -0.002640 | [-0.006273, +0.001345] |
| 1000 | G minus F | source_audit | -0.002749 | [-0.005260, -0.000298] |
| 1000 | D minus C | target_outer_val | +0.000331 | [-0.000077, +0.000747] |
| 1000 | D minus C | source_val | -0.015710 | [-0.022288, -0.010095] |
| 1000 | D minus C | source_audit | -0.014071 | [-0.019328, -0.008947] |

## Resource and supervision accounting

14 completed fits, 0.272 summed process-hours (including evaluation/I/O; not pure GPU kernel-hours).
Each fit has 100 target updates; preservation adds replay work recorded per run.
Nominal 200 actually labels 202 groups / 788 candidates (168 train + 34 validation groups).
Nominal 1,000 actually labels 1,001 groups / 3,848 candidates (839 train + 162 validation groups).
Source replay uses existing source supervision/our own teacher, not additional target labels.

## Source composition

| Surface | Study | Candidates | Groups |
|---|---|---:|---:|
| source_audit | deepprime | 11266 | 8697 |
| source_audit | hsu2026 | 9175 | 9175 |
| source_replay | deepprime | 2711 | 1710 |
| source_replay | hsu2026 | 4752 | 4752 |
| source_replay | pridict_pridict2 | 12047 | 4425 |
| source_val | deepprime | 492 | 367 |
| source_val | hsu2026 | 1986 | 1986 |
| source_val | pridict_pridict2 | 3279 | 1062 |
| target_budget_pool | deepprime | 10507 | 2853 |
| target_outer_val | deepprime | 23120 | 5561 |

See `screen_results.json` for all per-arm paired intervals, full precision and hashes.
These results do not establish a universal low-label threshold or reject all replay/residual methods.
