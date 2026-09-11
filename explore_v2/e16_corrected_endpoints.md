# E16 - the panel endpoints, recomputed with corrected accounting

Four defects in E11/E15's inline endpoint code were reproduced exactly and fixed in `endpoints.py`, whose invariants are pinned by 13 tests in `test_endpoints.py`. This is the corrected recomputation, printed beside the published values.

## What the recount changed

- panel rows: 118,187
- distinct (group, design) candidates after collapsing duplicate designs: 118,174 (13 rows collapsed)
- decision groups retained at depth >=2: 30,475 (published 30,475; +0)
- groups with a tied measured maximum: 5,808, of which 5,807 are all-zero

## Primary stratum, corrected

Each budget is scored on the groups that actually have that many distinct candidates, with that population's own oracle and random baseline:

| budget k | groups | oracle | random |
|---|---:|---:|---:|
| 1 | 30,475 | 0.0461 | 0.0217 |
| 3 | 18,660 | 0.0579 | 0.0469 |
| 5 | 8,775 | 0.0746 | 0.0674 |

| predictor | achieved @1 | @3 | @5 | hit rate @1 | regret @1 |
|---|---:|---:|---:|---:|---:|
| random choice | 0.0217 | 0.0469 | 0.0674 | 0.4442 | 0.02443 |
| OptiPrime | 0.0369 | 0.0558 | 0.0738 | 0.6647 | 0.00922 |
| PE-RankFormer, published member | 0.0363 | 0.0557 | 0.0738 | 0.6597 | 0.00986 |
| retrained, current ranking key | 0.0338 | 0.0548 | 0.0732 | 0.6286 | 0.01232 |
| retrained, canonical decision-group key | 0.0356 | 0.0555 | 0.0736 | 0.6457 | 0.01048 |

### Published versus corrected, the numbers that carry claims

| quantity | published | corrected |
|---|---:|---:|
| PE-RankFormer achieved @1 | 0.0363 | 0.0363 |
| OptiPrime achieved @1 | 0.0369 | 0.0369 |
| random achieved @1 | 0.0217 | 0.0217 |
| PE-RankFormer hit rate | 0.6599 | 0.6597 |
| OptiPrime hit rate | 0.6647 | 0.6647 |
| random hit rate | 0.3354 | 0.4442 |
| ours − OptiPrime @1 | -0.00062 | -0.00064 (CI [-0.00089, -0.00038], p 0.001) |
| canonical − control @1 | +0.00178 | +0.00184 (CI [+0.00158, +0.00210], p 0.001) |
| canonical − OptiPrime @1 | -0.00117 | -0.00126 (CI [-0.00153, -0.00100], p 0.001) |

Self-comparison control: a predictor against itself now returns observed 0.0, p = 1, degenerate flag True — where the published code returned p = 0.0005.


## informative groups

24,668 groups, 19,642 sites.

| predictor | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0268 | 0.3133 | 0.03018 |
| OptiPrime | 0.0456 | 0.5858 | 0.01139 |
| PE-RankFormer, published member | 0.0448 | 0.5795 | 0.01218 |
| retrained, current ranking key | 0.0418 | 0.5411 | 0.01522 |
| retrained, canonical decision-group key | 0.0440 | 0.5623 | 0.01295 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| ours − op | -0.00078 | [-0.00111, -0.00046] | 0.001 |
| e13_canon − op | -0.00156 | [-0.00189, -0.00122] | 0.001 |
| e13_canon − e13_ctrl | +0.00227 | [+0.00196, +0.00258] | 0.001 |

## depth 8+

2,400 groups, 2,371 sites.

| predictor | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0293 | 0.1130 | 0.06443 |
| OptiPrime | 0.0645 | 0.3617 | 0.02922 |
| PE-RankFormer, published member | 0.0620 | 0.3417 | 0.03171 |
| retrained, current ranking key | 0.0552 | 0.2850 | 0.03852 |
| retrained, canonical decision-group key | 0.0608 | 0.3279 | 0.03291 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| ours − op | -0.00249 | [-0.00416, -0.00083] | 0.004 |
| e13_canon − op | -0.00369 | [-0.00534, -0.00200] | 0.001 |
| e13_canon − e13_ctrl | +0.00561 | [+0.00402, +0.00727] | 0.001 |

## decision worth >= 0.05

5,614 groups, 5,286 sites.

| predictor | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0561 | 0.2197 | 0.08220 |
| OptiPrime | 0.1099 | 0.5891 | 0.02842 |
| PE-RankFormer, published member | 0.1066 | 0.5549 | 0.03171 |
| retrained, current ranking key | 0.0972 | 0.4797 | 0.04111 |
| retrained, canonical decision-group key | 0.1045 | 0.5328 | 0.03382 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| ours − op | -0.00329 | [-0.00451, -0.00209] | 0.001 |
| e13_canon − op | -0.00540 | [-0.00672, -0.00410] | 0.001 |
| e13_canon − e13_ctrl | +0.00729 | [+0.00609, +0.00853] | 0.001 |