# E02 - the frozen predictions scored on the user's decision

Descriptive re-scoring of frozen predictions. No fitting. Intervals are protospacer-clustered percentile bootstraps.

The five metadata-identical replicate groups inside fold 0 are averaged first, taking 20,509 rows to 20,441 candidate measurements: a repeat measurement is not an alternative design.

## 1. The grouping ladder

| grouping | eligible groups | OptiPrime | ordinal-S4D | final ensemble | delta (ens - OP) |
|---|---:|---:|---:|---:|---:|
| spacer (manuscript) (>= 5) | 670 | 0.5471 | 0.6353 | 0.6357 | +0.0885 |
| spacer x source/cell/editor (>= 5) | 988 | 0.4606 | 0.5845 | 0.5955 | +0.1349 |
| allele x context, >=3 designs (>= 3) | 73 | 0.6252 | 0.6132 | 0.6512 | +0.0260 |
| allele x context, >=2 designs (>= 2) | 1,463 | 0.3292 | 0.5063 | 0.5028 | +0.1735 |

A protospacer group of >=5 rows contains a median of 1 intended alleles across 10 conditions, i.e. a median of 1.8 rows per allele-and-condition cell. 100% of those groups mix conditions and 26% mix alleles. Ranking inside such a group is mostly the question *which edit, in which cell line, is easier* - not *which pegRNA should I order*.

## 2. The decision fold 0 can actually score: pick one of two designs

2,327 decision groups on fold 0 hold exactly two alternative designs for one canonical allele in one fully specified context, spanning 564 protospacers. 937 are exact ties in the measured outcome (937 of them both-zero) and cannot be scored, leaving **1,390 scorable binary choices**.

| predictor | accuracy | efficiency of the pick | regret |
|---|---:|---:|---:|
| random choice | 0.500 | 0.1092 | 0.0316 |
| OptiPrime | 0.657 | 0.1252 | 0.0156 |
| ordinal-S4D member | 0.750 | 0.1340 | 0.0068 |
| PE-RankFormer (final ensemble) | 0.747 | 0.1343 | 0.0065 |
| oracle (best of the two) | 1.000 | 0.1408 | 0.0000 |

| paired difference | value | 95% CI | p |
|---|---:|---|---:|
| ours minus op correct | +0.0906 | [+0.0601, +0.1208] | 0.0005 |
| ordssm minus op correct | +0.0935 | [+0.0586, +0.1265] | 0.0005 |
| ours minus op picked | +0.0091 | [+0.0063, +0.0124] | 0.0005 |
| ordssm minus op picked | +0.0088 | [+0.0057, +0.0123] | 0.0005 |
| ours minus op regret | -0.0091 | [-0.0124, -0.0063] | 0.0005 |
| ordssm minus op regret | -0.0088 | [-0.0123, -0.0057] | 0.0005 |

The whole decision is worth 0.0632 efficiency on average (median gap 0.0169); the better of the two designs delivers 0.1408. That is the ceiling any pegRNA-selection method can win here, and it bounds how much a ranking improvement can be worth at the bench.

### By how much the two designs actually differ

| measured gap | groups | mean gap | OP acc | S4D acc | ens acc | ens regret |
|---|---:|---:|---:|---:|---:|---:|
| [0.00, 0.01) | 610 | 0.0026 | 0.551 | 0.656 | 0.628 | 0.0007 |
| [0.01, 0.05) | 341 | 0.0264 | 0.710 | 0.751 | 0.765 | 0.0063 |
| [0.05, 0.20) | 307 | 0.1137 | 0.779 | 0.863 | 0.893 | 0.0112 |
| [0.20, 1.01) | 132 | 0.3204 | 0.727 | 0.924 | 0.917 | 0.0223 |

### By condition

| condition | groups | mean gap | OP acc | S4D acc | ens acc |
|---|---:|---:|---:|---:|---:|
| deepprime|HEK293T|PE2 | 404 | 0.0693 | 0.730 | 0.782 | 0.760 |
| deepprime|A549|PE4 | 294 | 0.0470 | 0.452 | 0.684 | 0.653 |
| deepprime|A549|PE2 | 285 | 0.0421 | 0.758 | 0.796 | 0.796 |
| deepprime|HEK293T|PE4 | 281 | 0.0698 | 0.541 | 0.658 | 0.705 |
| deepprime|DLD1|PE4 | 35 | 0.1954 | 0.971 | 0.914 | 0.914 |

### By whether training saw the same decision group

| decision group also in training | groups | OP acc | ens acc | ens regret |
|---|---:|---:|---:|---:|
| False | 686 | 0.602 | 0.713 | 0.0060 |
| True | 704 | 0.710 | 0.781 | 0.0069 |