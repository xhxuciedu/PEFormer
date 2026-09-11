# E13 - the ranking loss is grouped on a design artefact

`ranking_group_key` builds the pairwise ranking loss's groups from the raw `full_unedited | full_edited` string pair plus cell type and editor. That window's extent depends on the row's RTT length, and alternative pegRNAs for one allele differ precisely in PBS/RTT geometry -- so the key meant to group them together is the one most likely to split them apart.

> **What this table is and is not.** These are *possible* within-group design pairs. Measured exposure is a different and smaller thing, and E17a replays the real sampler to get it: 15,695 sampled pair instances per epoch under the current key against 62,028 under the canonical one (3.95x), with the share of training groups receiving any ranking update rising from 3.2% to 19.1%. The sharpest measured fact is that **100.0% of the current key's sampled pairs compare two designs of identical RTT length**, against 6.8% under the canonical key: for one allele the window's extent is set by the RTT, so an identical window pair implies an identical RTT length, and the ranking term has never once been asked to compare the geometry variation that distinguishes alternative pegRNAs.

> **A confound in the arms below.** `corpus.group_key` is passed both to `GroupedBatchSampler` and to `sample_ranking_pairs`, so changing it changes batch composition *and* pair eligibility. E14/E15 therefore measure the combined effect, not the ranking channel alone.

Measured on the 238,381 rows the model trains on (folds 2-5):

| grouping | groups | singletons | rows in a multi-design group | within-group design pairs |
|---|---:|---:|---:|---:|
| current ranking key | 209,161 | 93.8% | 16.2% | 14,277 |
| canonical decision group | 132,066 | 68.0% | 62.3% | **368,307** |

The ranking term, whose loss coefficient is 0.25, has had access to **3.9%** of the *possible* within-group design comparisons -- a factor of 25.8 fewer. Measured exposure differs by 3.95x rather than 25.8x; both are reported because the first bounds the opportunity and the second is what the objective actually consumed.

Designs per group, current key: {'1': 196098, '2': 12456, '3': 607}; canonical: {'1': 89828, '2': 27605, '3': 1848, '4': 5636, '5': 225, '6': 611}
