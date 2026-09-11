# E05 - reference-panel adaptation on out-of-fold development episodes

Frozen out-of-fold ordinal-S4D backbone. Support and query alleles are disjoint; every arm receives the same support labels; the PCA basis and (lambda, rank) come from source contexts only. Fold 0 is not used.

297,755 development candidate measurements, 20 contexts, budgets [0, 12, 24, 48, 96, 192], 5 support draws each.

## Selection quality by measurement budget

Mean over episodes of within-decision-group top-1 accuracy. **Read the paired table below instead for the comparison**: the rank is chosen per target context, so `lowrank2` and `lowrank4` are averaged over different context sets and their marginal means are not comparable with each other.

| arm | B=0 | B=12 | B=24 | B=48 | B=96 | B=192 |
|---|---|---|---|---|---|---|
| base | 0.652 | 0.657 | 0.655 | 0.659 | 0.655 | 0.654 |
| intercept | - | 0.657 | 0.655 | 0.659 | 0.655 | 0.654 |
| affine | - | 0.650 | 0.655 | 0.659 | 0.655 | 0.654 |
| lowrank2 | - | 0.632 | 0.619 | 0.623 | 0.616 | 0.613 |
| lowrank4 | - | 0.680 | 0.674 | 0.671 | 0.673 | 0.671 |
| residual_ridge | - | 0.639 | 0.630 | 0.630 | 0.630 | 0.628 |
| support_only_ridge | - | 0.535 | 0.549 | 0.552 | 0.548 | 0.551 |
| shuffled_support | - | 0.653 | 0.652 | 0.651 | 0.651 | 0.650 |
| within_lowrank2 | - | 0.646 | 0.639 | 0.636 | 0.630 | 0.619 |
| within_lowrank4 | - | 0.682 | 0.676 | 0.674 | 0.677 | 0.676 |
| within_ridge | - | 0.654 | 0.645 | 0.642 | 0.638 | 0.628 |
| within_ridge_shuffled | - | 0.655 | 0.654 | 0.655 | 0.650 | 0.648 |

## Regret by measurement budget

| arm | B=0 | B=12 | B=24 | B=48 | B=96 | B=192 |
|---|---|---|---|---|---|---|
| base | 0.0168 | 0.0153 | 0.0158 | 0.0155 | 0.0154 | 0.0160 |
| intercept | - | 0.0153 | 0.0158 | 0.0155 | 0.0154 | 0.0160 |
| affine | - | 0.0160 | 0.0158 | 0.0155 | 0.0154 | 0.0160 |
| lowrank2 | - | 0.0169 | 0.0178 | 0.0172 | 0.0171 | 0.0178 |
| lowrank4 | - | 0.0117 | 0.0126 | 0.0129 | 0.0122 | 0.0126 |
| residual_ridge | - | 0.0160 | 0.0170 | 0.0167 | 0.0163 | 0.0170 |
| support_only_ridge | - | 0.0340 | 0.0325 | 0.0327 | 0.0326 | 0.0321 |
| shuffled_support | - | 0.0157 | 0.0161 | 0.0160 | 0.0157 | 0.0163 |
| within_lowrank2 | - | 0.0163 | 0.0171 | 0.0171 | 0.0169 | 0.0176 |
| within_lowrank4 | - | 0.0118 | 0.0126 | 0.0127 | 0.0123 | 0.0127 |
| within_ridge | - | 0.0154 | 0.0163 | 0.0163 | 0.0160 | 0.0168 |
| within_ridge_shuffled | - | 0.0156 | 0.0161 | 0.0159 | 0.0157 | 0.0162 |

## Paired against no adaptation, within episode

| arm | budget | episodes | delta accuracy | SD across episodes | delta regret | episodes improved |
|---|---:|---:|---:|---:|---:|---:|
| intercept | 12 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| intercept | 24 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| intercept | 48 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| intercept | 96 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| intercept | 192 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| affine | 12 | 80 | -0.0068 | 0.0511 | +0.00065 | 0/80 |
| affine | 24 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| affine | 48 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| affine | 96 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| affine | 192 | 80 | +0.0000 | 0.0000 | +0.00000 | 0/80 |
| lowrank2 | 12 | 65 | -0.0197 | 0.0270 | +0.00082 | 8/65 |
| lowrank2 | 24 | 65 | -0.0309 | 0.0313 | +0.00124 | 3/65 |
| lowrank2 | 48 | 65 | -0.0329 | 0.0336 | +0.00114 | 4/65 |
| lowrank2 | 96 | 65 | -0.0329 | 0.0307 | +0.00105 | 4/65 |
| lowrank2 | 192 | 65 | -0.0346 | 0.0309 | +0.00109 | 5/65 |
| lowrank4 | 12 | 15 | -0.0026 | 0.0035 | -0.00029 | 2/15 |
| lowrank4 | 24 | 15 | -0.0021 | 0.0049 | -0.00029 | 4/15 |
| lowrank4 | 48 | 15 | -0.0037 | 0.0065 | -0.00003 | 1/15 |
| lowrank4 | 96 | 15 | -0.0052 | 0.0043 | -0.00023 | 0/15 |
| lowrank4 | 192 | 15 | -0.0075 | 0.0085 | -0.00037 | 2/15 |
| residual_ridge | 12 | 80 | -0.0186 | 0.0256 | +0.00067 | 9/80 |
| residual_ridge | 24 | 80 | -0.0250 | 0.0297 | +0.00120 | 8/80 |
| residual_ridge | 48 | 80 | -0.0293 | 0.0341 | +0.00121 | 8/80 |
| residual_ridge | 96 | 80 | -0.0247 | 0.0292 | +0.00085 | 11/80 |
| residual_ridge | 192 | 80 | -0.0258 | 0.0327 | +0.00094 | 14/80 |
| support_only_ridge | 12 | 80 | -0.1225 | 0.1581 | +0.01866 | 20/80 |
| support_only_ridge | 24 | 80 | -0.1060 | 0.1552 | +0.01669 | 27/80 |
| support_only_ridge | 48 | 80 | -0.1075 | 0.1525 | +0.01718 | 22/80 |
| support_only_ridge | 96 | 80 | -0.1064 | 0.1520 | +0.01721 | 24/80 |
| support_only_ridge | 192 | 80 | -0.1032 | 0.1468 | +0.01612 | 24/80 |
| shuffled_support | 12 | 80 | -0.0046 | 0.0194 | +0.00034 | 24/80 |
| shuffled_support | 24 | 80 | -0.0032 | 0.0127 | +0.00028 | 25/80 |
| shuffled_support | 48 | 80 | -0.0082 | 0.0175 | +0.00052 | 18/80 |
| shuffled_support | 96 | 80 | -0.0032 | 0.0170 | +0.00033 | 26/80 |
| shuffled_support | 192 | 80 | -0.0038 | 0.0168 | +0.00024 | 31/80 |
| within_lowrank2 | 12 | 65 | -0.0054 | 0.0140 | +0.00019 | 14/65 |
| within_lowrank2 | 24 | 65 | -0.0115 | 0.0174 | +0.00060 | 13/65 |
| within_lowrank2 | 48 | 65 | -0.0197 | 0.0217 | +0.00104 | 2/65 |
| within_lowrank2 | 96 | 65 | -0.0193 | 0.0270 | +0.00077 | 6/65 |
| within_lowrank2 | 192 | 65 | -0.0293 | 0.0305 | +0.00089 | 5/65 |
| within_lowrank4 | 12 | 15 | -0.0006 | 0.0027 | -0.00010 | 3/15 |
| within_lowrank4 | 24 | 15 | -0.0001 | 0.0033 | -0.00024 | 2/15 |
| within_lowrank4 | 48 | 15 | -0.0007 | 0.0037 | -0.00021 | 2/15 |
| within_lowrank4 | 96 | 15 | -0.0008 | 0.0035 | -0.00019 | 3/15 |
| within_lowrank4 | 192 | 15 | -0.0025 | 0.0040 | -0.00021 | 1/15 |
| within_ridge | 12 | 80 | -0.0035 | 0.0166 | +0.00011 | 24/80 |
| within_ridge | 24 | 80 | -0.0098 | 0.0162 | +0.00049 | 13/80 |
| within_ridge | 48 | 80 | -0.0175 | 0.0213 | +0.00079 | 9/80 |
| within_ridge | 96 | 80 | -0.0166 | 0.0292 | +0.00054 | 12/80 |
| within_ridge | 192 | 80 | -0.0255 | 0.0303 | +0.00078 | 7/80 |
| within_ridge_shuffled | 12 | 80 | -0.0024 | 0.0117 | +0.00027 | 21/80 |
| within_ridge_shuffled | 24 | 80 | -0.0012 | 0.0146 | +0.00029 | 30/80 |
| within_ridge_shuffled | 48 | 80 | -0.0040 | 0.0152 | +0.00039 | 22/80 |
| within_ridge_shuffled | 96 | 80 | -0.0043 | 0.0186 | +0.00029 | 28/80 |
| within_ridge_shuffled | 192 | 80 | -0.0058 | 0.0177 | +0.00022 | 20/80 |

A per-context intercept cannot change the order of candidates inside a decision group, so its row is exactly zero at every budget. That is an arithmetic check on the pipeline, not a result: **a calibration success is not a design-transfer success.** The affine arm is exactly zero too wherever its fitted slope is positive, and differs from the base at B=12 only because in one episode of eighty the least-squares slope on twelve support points came out negative and inverted the ranking. That is worth keeping in view: at the smallest budgets even a two-parameter recalibration can actively damage a ranking it was supposed to leave alone.
