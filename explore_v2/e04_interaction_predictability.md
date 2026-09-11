# E04 - is the interaction predictable on held-out loci?

111,862 matched quartets over 10,103 locus groups and 24 contexts; SD of D on the arcsine scale is 0.2004. Locus-grouped 5-fold, ridge penalty chosen inside each training split, no intercept (D is antisymmetric).

## Held-out prediction of D

| model | Spearman | Pearson | R^2 vs predicting zero | sign accuracy |
|---|---:|---:|---:|---:|
| additive_predict_zero | n/a | n/a | +0.0000 | 0.500 |
| feature_x_context | 0.3232 | 0.3239 | +0.1058 | 0.509 |
| embedding_x_context | 0.4901 | 0.6254 | +0.3915 | 0.547 |
| design_features_only | 0.0758 | 0.0485 | +0.0030 | 0.440 |
| feature_x_context_SHUFFLED_context | 0.0957 | 0.1051 | +0.0119 | 0.441 |
| feature_x_context_SHUFFLED_D | 0.0688 | 0.0788 | +0.0064 | 0.415 |

### Where the signal sits

Sign agreement for `embedding_x_context`, restricted to the largest quartets by true \|D\| and by predicted \|D\|.

| subset | sign accuracy |
|---|---:|
| sign acc by true top 50pct | 0.685 |
| sign acc by pred top 50pct | 0.697 |
| sign acc by true top 19pct | 0.803 |
| sign acc by pred top 19pct | 0.852 |
| sign acc by true top 9pct | 0.864 |
| sign acc by pred top 9pct | 0.916 |
| sign acc by true top 5pct | 0.908 |
| sign acc by pred top 5pct | 0.938 |

## Injected-interaction positive control

Same features, same splits, same penalty search, with a synthetic feature-by-context interaction added at a known size.

| injected SD (fraction of SD of D) | Spearman | R^2 | sign accuracy |
|---|---:|---:|---:|
| 0.05 | 0.3339 | +0.1129 | 0.565 |
| 0.10 | 0.3458 | +0.1218 | 0.571 |
| 0.25 | 0.3875 | +0.1702 | 0.586 |
| 0.50 | 0.4561 | +0.2988 | 0.608 |

The positive control says what effect size this analysis could have found. A null result above is informative only down to the smallest injected size the pipeline still recovers.
