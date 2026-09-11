# E09 - E04's claim retested with one coordinate system and no encoder exposure

47,470 of 110,321 quartets have all four measurements inside a single official fold. Those are the only ones where both designs are embedded by the same checkpoint **and** that checkpoint never trained on any row in the analysis. Everything below uses only those, fold by fold, with locus-grouped CV and every transform fitted inside the training split.

## Held-out prediction of D, per fold

| fold | quartets | loci | contexts | embedding x context | design only | SHUFFLED context | SHUFFLED D |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (historical held-out) | 3,509 | 508 | 14 | -2.8178 | -0.1073 | -0.0388 | -2.4815 |
| 1 | 8,651 | 1,985 | 24 | +0.2249 | +0.0647 | +0.0818 | -0.0008 |
| 2 | 8,695 | 2,008 | 24 | -0.0558 | +0.0554 | +0.0664 | -0.0004 |
| 3 | 9,033 | 2,031 | 24 | +0.2719 | +0.0775 | +0.0589 | +0.0025 |
| 4 | 8,694 | 2,041 | 24 | +0.1790 | +0.0567 | +0.0864 | -0.0194 |
| 5 | 8,888 | 2,023 | 24 | +0.2990 | +0.0772 | +0.0925 | -0.0073 |

R² against predicting zero. All four arms share the representation, the PCA dimension, the penalty grid and the split.

## Pooled over development folds

| arm | mean R² | range | mean Spearman |
|---|---:|---|---:|
| embedding_x_context | +0.1838 | [-0.0558, +0.2990] | 0.4183 |
| embedding_design_only | +0.0663 | [+0.0554, +0.0775] | 0.1758 |
| SHUFFLED_context_same_features | +0.0772 | [+0.0589, +0.0925] | 0.1931 |
| SHUFFLED_D_same_features | -0.0051 | [-0.0194, +0.0025] | 0.0544 |

## Sign of D versus actual order reversal

Section 2.6 of the next-steps plan is right that these are different events: D can be large with both contrasts the same sign, which is a change of margin and not a change of choice. Reported separately, on development folds.

| fold | sign of D, all | sign of D, top 5% | supported quartets in top 5% | reversal rate there | reversal balanced accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 0.544 | 0.838 | 214 | 0.023 | 0.763 |
| 2 | 0.546 | 0.701 | 180 | 0.033 | 0.747 |
| 3 | 0.558 | 0.845 | 209 | 0.057 | 0.820 |
| 4 | 0.536 | 0.763 | 184 | 0.038 | 0.864 |
| 5 | 0.551 | 0.899 | 233 | 0.000 | 0.358 |

## Positive control, injected into a null target

E04's control added a synthetic signal to the observed D and scored the sum, which cannot show that the added component was the part recovered. Here the signal is injected into pure measurement noise at the additive-model scale, and the recovery of that component is measured directly.

| injected SD (fraction of SD of D) | R² on the injected target | correlation with the injected component |
|---|---:|---:|
| 0.02 | +0.0010 | 0.402 |
| 0.05 | +0.0029 | 0.521 |
| 0.10 | +0.0316 | 0.764 |
| 0.25 | +0.1856 | 0.845 |