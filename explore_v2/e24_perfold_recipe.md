# E24 - the improved recipe, built the way the published model is built

> **Disclosure.** A further development use of the reserved panel; nothing here is confirmatory.

Five checkpoints, one per official fold, combined by plain mean of predicted efficiencies, matching the published member. Calibrator: isotonic, fitted on this model's own development out-of-fold predictions (297,962 rows over folds 1-5), frozen before either surface was scored. Fold 0 is the official test fold and no member trains or early-stops on it.

## panel - all eligible groups: 30,475 groups, 22,931 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0217 | 0.4442 | 0.02443 |
| OptiPrime | 0.0369 | 0.6647 | 0.00922 |
| PE-RankFormer, published member (5 per-fold) | 0.0363 | 0.6597 | 0.00986 |
| arm A, 3 seeds (released recipe) | 0.0349 | 0.6405 | 0.01120 |
| arm F, 3 same-data seeds (E22) | 0.0361 | 0.6539 | 0.01005 |
| **arm F, 5 per-fold (this experiment)** | 0.0361 | 0.6571 | 0.01003 |
| perfect chooser | 0.0461 | 1.0000 | 0.00000 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| **arm F, 5 per-fold (this experiment)** − PE-RankFormer, published member (5 per-fold) | -0.00017 | [-0.00034, +0.00001] | 0.062 |
| **arm F, 5 per-fold (this experiment)** − OptiPrime | -0.00081 | [-0.00107, -0.00055] | 0.001 |
| **arm F, 5 per-fold (this experiment)** − arm F, 3 same-data seeds (E22) | +0.00002 | [-0.00015, +0.00020] | 0.812 |
| **arm F, 5 per-fold (this experiment)** − arm A, 3 seeds (released recipe) | +0.00117 | [+0.00095, +0.00140] | 0.001 |
| **arm F, 5 per-fold (this experiment)** − E_ens | -0.00023 | [-0.00043, -0.00004] | 0.014 |
| **arm F, 5 per-fold (this experiment)** − H_ens | -0.00041 | [-0.00059, -0.00023] | 0.001 |
| PE-RankFormer, published member (5 per-fold) − OptiPrime | -0.00064 | [-0.00089, -0.00038] | 0.001 |

## panel - informative groups: 24,668 groups, 19,642 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0268 | 0.3133 | 0.03018 |
| OptiPrime | 0.0456 | 0.5858 | 0.01139 |
| PE-RankFormer, published member (5 per-fold) | 0.0448 | 0.5795 | 0.01218 |
| arm A, 3 seeds (released recipe) | 0.0431 | 0.5558 | 0.01384 |
| arm F, 3 same-data seeds (E22) | 0.0446 | 0.5724 | 0.01242 |
| **arm F, 5 per-fold (this experiment)** | 0.0446 | 0.5764 | 0.01239 |
| perfect chooser | 0.0570 | 1.0000 | 0.00000 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| **arm F, 5 per-fold (this experiment)** − PE-RankFormer, published member (5 per-fold) | -0.00021 | [-0.00042, +0.00001] | 0.074 |
| **arm F, 5 per-fold (this experiment)** − OptiPrime | -0.00100 | [-0.00131, -0.00067] | 0.001 |
| **arm F, 5 per-fold (this experiment)** − arm F, 3 same-data seeds (E22) | +0.00003 | [-0.00018, +0.00024] | 0.775 |
| **arm F, 5 per-fold (this experiment)** − arm A, 3 seeds (released recipe) | +0.00145 | [+0.00119, +0.00172] | 0.001 |
| **arm F, 5 per-fold (this experiment)** − E_ens | -0.00029 | [-0.00051, -0.00004] | 0.014 |
| **arm F, 5 per-fold (this experiment)** − H_ens | -0.00050 | [-0.00073, -0.00028] | 0.001 |
| PE-RankFormer, published member (5 per-fold) − OptiPrime | -0.00078 | [-0.00111, -0.00046] | 0.001 |

## panel - depth 8+: 2,400 groups, 2,371 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0293 | 0.1130 | 0.06443 |
| OptiPrime | 0.0645 | 0.3617 | 0.02922 |
| PE-RankFormer, published member (5 per-fold) | 0.0620 | 0.3417 | 0.03171 |
| arm A, 3 seeds (released recipe) | 0.0573 | 0.2954 | 0.03639 |
| arm F, 3 same-data seeds (E22) | 0.0618 | 0.3279 | 0.03190 |
| **arm F, 5 per-fold (this experiment)** | 0.0623 | 0.3379 | 0.03147 |
| perfect chooser | 0.0937 | 1.0000 | 0.00000 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| **arm F, 5 per-fold (this experiment)** − PE-RankFormer, published member (5 per-fold) | +0.00024 | [-0.00090, +0.00136] | 0.690 |
| **arm F, 5 per-fold (this experiment)** − OptiPrime | -0.00224 | [-0.00393, -0.00068] | 0.004 |
| **arm F, 5 per-fold (this experiment)** − arm F, 3 same-data seeds (E22) | +0.00044 | [-0.00072, +0.00160] | 0.470 |
| **arm F, 5 per-fold (this experiment)** − arm A, 3 seeds (released recipe) | +0.00492 | [+0.00352, +0.00643] | 0.001 |
| **arm F, 5 per-fold (this experiment)** − E_ens | -0.00006 | [-0.00126, +0.00116] | 0.951 |
| **arm F, 5 per-fold (this experiment)** − H_ens | -0.00126 | [-0.00233, -0.00012] | 0.026 |
| PE-RankFormer, published member (5 per-fold) − OptiPrime | -0.00249 | [-0.00416, -0.00083] | 0.004 |

## panel - decision worth >= 0.05: 5,614 groups, 5,286 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0561 | 0.2197 | 0.08220 |
| OptiPrime | 0.1099 | 0.5891 | 0.02842 |
| PE-RankFormer, published member (5 per-fold) | 0.1066 | 0.5549 | 0.03171 |
| arm A, 3 seeds (released recipe) | 0.1015 | 0.5110 | 0.03680 |
| arm F, 3 same-data seeds (E22) | 0.1060 | 0.5435 | 0.03235 |
| **arm F, 5 per-fold (this experiment)** | 0.1060 | 0.5465 | 0.03233 |
| perfect chooser | 0.1383 | 1.0000 | 0.00000 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| **arm F, 5 per-fold (this experiment)** − PE-RankFormer, published member (5 per-fold) | -0.00062 | [-0.00145, +0.00022] | 0.155 |
| **arm F, 5 per-fold (this experiment)** − OptiPrime | -0.00391 | [-0.00512, -0.00266] | 0.001 |
| **arm F, 5 per-fold (this experiment)** − arm F, 3 same-data seeds (E22) | +0.00002 | [-0.00084, +0.00088] | 0.935 |
| **arm F, 5 per-fold (this experiment)** − arm A, 3 seeds (released recipe) | +0.00447 | [+0.00340, +0.00554] | 0.001 |
| **arm F, 5 per-fold (this experiment)** − E_ens | -0.00097 | [-0.00188, -0.00005] | 0.039 |
| **arm F, 5 per-fold (this experiment)** − H_ens | -0.00196 | [-0.00282, -0.00111] | 0.001 |
| PE-RankFormer, published member (5 per-fold) − OptiPrime | -0.00329 | [-0.00451, -0.00209] | 0.001 |

## correlations - reserved panel

| model | Spearman | Pearson |
|---|---:|---:|
| OptiPrime | 0.6931 | 0.5854 |
| PE-RankFormer, published member (5 per-fold) | 0.7091 | 0.5522 |
| **arm F, 5 per-fold (this experiment)** | 0.7096 | 0.5572 |
| arm F, 5 per-fold, calibrated | 0.7096 | 0.6048 |

| margin | value | 95% CI | p |
|---|---:|---|---:|
| spearman Ffold | +0.0164 | [+0.0141, +0.0188] | 0.0005 |
| pearson Ffold calibrated | +0.0194 | [+0.0157, +0.0230] | 0.0005 |
| spearman Ffold minus published | +0.0005 | [-0.0004, +0.0014] | 0.2830 |

## correlations - held-out fold 0

| model | Spearman | Pearson |
|---|---:|---:|
| OptiPrime | 0.8690 | 0.8270 |
| PE-RankFormer, published member (5 per-fold) | 0.9079 | 0.7278 |
| **arm F, 5 per-fold (this experiment)** | 0.9084 | 0.7843 |
| arm F, 5 per-fold, calibrated | 0.9084 | 0.8637 |

| margin | value | 95% CI | p |
|---|---:|---|---:|
| spearman Ffold | +0.0394 | [+0.0293, +0.0502] | 0.0005 |
| pearson Ffold calibrated | +0.0367 | [+0.0194, +0.0575] | 0.0005 |

## fold 0 - the decision: 1,463 informative groups, 439 clusters

| model | achieved @1 |
|---|---:|
| OptiPrime | 0.12721 |
| PE-RankFormer, published member (5 per-fold) | 0.13636 |
| **arm F, 5 per-fold (this experiment)** | 0.13644 |
| random choice | 0.10856 |
| perfect chooser | 0.14304 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| **arm F, 5 per-fold (this experiment)** − PE-RankFormer, published member (5 per-fold) | +0.00007 | [-0.00152, +0.00173] | 0.952 |
| **arm F, 5 per-fold (this experiment)** − OptiPrime | +0.00922 | [+0.00612, +0.01255] | 0.001 |
| PE-RankFormer, published member (5 per-fold) − OptiPrime | +0.00915 | [+0.00633, +0.01215] | 0.001 |