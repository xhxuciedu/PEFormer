# E25 - the adaptation partition

> **Disclosure.** The panel already informed E11-E24. This partition prevents new training leakage and supports a retrospective adaptation evaluation; it is not a pristine confirmatory surface and the no-target-label comparison remains the manuscript's claim.

## Why components rather than groups

13,803 alleles are reachable by more than one protospacer and 9,893 protospacers install more than one allele, so alleles and protospacers are not independent units. Their bipartite graph has **20,276 connected components** over 30,475 alleles and 30,302 protospacers; whole components are assigned to a split.

Component size is heavily skewed: median 1 group(s), mean 1.50, 76.1% singletons, largest 30 groups.

## Realised split

| | components | groups | eligible (d$\geq$2) | informative | d$\geq$5 | candidates | measurements | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10,370 | 19,357 | 19,357 | 15,337 | 4,658 | 72,010 | 72,014 | 63.5% |
| val | 4,952 | 5,561 | 5,561 | 4,656 | 2,056 | 23,120 | 23,126 | 18.2% |
| test | 4,954 | 5,557 | 5,557 | 4,675 | 2,061 | 23,044 | 23,047 | 18.2% |

## The splits are comparable

| split | mean efficiency | zero fraction | mean PBS | mean RTT |
|---|---:|---:|---:|---:|
| train | 0.0233 | 0.392 | 9.65 | 24.45 |
| val | 0.0258 | 0.368 | 9.62 | 24.64 |
| test | 0.0253 | 0.371 | 9.64 | 24.55 |

The assignment balances two quantities at once: total groups, which sets the headline allocation, and groups of depth five or more, which are where the decision is hardest and where a few large components could otherwise absorb all the evaluation power. Balancing both trades a little accuracy on the 60/20/20 group target for a usable spread of deep groups; realised shares are in the table above rather than assumed.

Written to `explore_v2/cache/adaptation_partition_v2.parquet`. Model selection uses **validation only**; the test components are not consulted while choosing architectures, losses, fine-tuning depth or stopping epochs.
