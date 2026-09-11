# E14 - fixing the ranking loss's grouping

Surface: locked fold 0 (20,509 rows); neither arm trains on it or early-stops on it. Same architecture, same seed, same code; the arms differ only in which rows the pairwise ranking loss may compare.

## Pooled Spearman, the manuscript's headline metric

| arm | pooled rho |
|---|---:|
| control (current ranking key) | 0.8927 |
| canonical decision-group key | 0.8952 |

## The fixed-allele decision the ranking term is meant to serve

1,463 scorable decision groups on fold 0 (oracle 0.1432, random 0.1086).

| arm | accuracy | achieved efficiency | regret | share of random regret removed |
|---|---:|---:|---:|---:|
| control (current ranking key) | 0.7211 | 0.1330 | 0.01018 | 70.6% |
| canonical decision-group key | 0.7239 | 0.1356 | 0.00759 | 78.1% |

## Paired, clustered on protospacer

| difference (canonical key - control) | value | 95% CI | p |
|---|---:|---|---:|
| best-design accuracy | +0.00273 | [-0.02081, +0.02565] | 0.894 |
| achieved efficiency | +0.00258 | [+0.00046, +0.00530] | 0.01 |

1,463 groups over 439 protospacer clusters.

Fold 0 has been examined many times across nine rounds, so this is development evidence about a training choice, not a confirmatory result. The reserved panel is deliberately not used: it was spent on its one pre-declared comparison in E11.
