# E22 - the best training recipe, on the panel

> **Disclosure.** A further development use of the reserved panel; nothing here is confirmatory.

Arms A and F are each rank-averaged over their three seeds, so ensemble size is matched between them and cannot explain a difference.

## all eligible groups — 30,475 groups, 22,931 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0217 | 0.4442 | 0.02443 |
| OptiPrime | 0.0369 | 0.6647 | 0.00922 |
| PE-RankFormer, published member | 0.0363 | 0.6597 | 0.00986 |
| arm A, 3-seed ensemble (released recipe) | 0.0349 | 0.6405 | 0.01120 |
| arm F, 3-seed ensemble (canonical batching + features) | 0.0361 | 0.6539 | 0.01005 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| arm A, 3-seed ensemble (released recipe) − OptiPrime | -0.00198 | [-0.00226, -0.00169] | 0.001 |
| D_ens − OptiPrime | -0.00070 | [-0.00095, -0.00045] | 0.001 |
| E_ens − OptiPrime | -0.00057 | [-0.00083, -0.00034] | 0.001 |
| arm F, 3-seed ensemble (canonical batching + features) − OptiPrime | -0.00083 | [-0.00109, -0.00057] | 0.001 |
| H_ens − OptiPrime | -0.00040 | [-0.00065, -0.00014] | 0.001 |
| arm A, 3-seed ensemble (released recipe) − PE-RankFormer, published member | -0.00134 | [-0.00157, -0.00112] | 0.001 |
| D_ens − PE-RankFormer, published member | -0.00006 | [-0.00024, +0.00012] | 0.517 |
| E_ens − PE-RankFormer, published member | +0.00006 | [-0.00011, +0.00023] | 0.464 |
| arm F, 3-seed ensemble (canonical batching + features) − PE-RankFormer, published member | -0.00019 | [-0.00038, -0.00000] | 0.044 |
| H_ens − PE-RankFormer, published member | +0.00024 | [+0.00005, +0.00043] | 0.02 |
| arm F, 3-seed ensemble (canonical batching + features) − arm A, 3-seed ensemble (released recipe) | +0.00115 | [+0.00094, +0.00138] | 0.001 |
| arm F, 3-seed ensemble (canonical batching + features) − E_ens | -0.00025 | [-0.00045, -0.00007] | 0.006 |
| H_ens − E_ens | +0.00018 | [-0.00001, +0.00036] | 0.068 |
| E_ens − arm A, 3-seed ensemble (released recipe) | +0.00141 | [+0.00119, +0.00163] | 0.001 |
| PE-RankFormer, published member − OptiPrime | -0.00064 | [-0.00089, -0.00038] | 0.001 |

## informative groups — 24,668 groups, 19,642 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0268 | 0.3133 | 0.03018 |
| OptiPrime | 0.0456 | 0.5858 | 0.01139 |
| PE-RankFormer, published member | 0.0448 | 0.5795 | 0.01218 |
| arm A, 3-seed ensemble (released recipe) | 0.0431 | 0.5558 | 0.01384 |
| arm F, 3-seed ensemble (canonical batching + features) | 0.0446 | 0.5724 | 0.01242 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| arm A, 3-seed ensemble (released recipe) − OptiPrime | -0.00245 | [-0.00280, -0.00211] | 0.001 |
| D_ens − OptiPrime | -0.00086 | [-0.00117, -0.00055] | 0.001 |
| E_ens − OptiPrime | -0.00071 | [-0.00103, -0.00040] | 0.001 |
| arm F, 3-seed ensemble (canonical batching + features) − OptiPrime | -0.00102 | [-0.00134, -0.00070] | 0.001 |
| H_ens − OptiPrime | -0.00049 | [-0.00081, -0.00017] | 0.001 |
| arm A, 3-seed ensemble (released recipe) − PE-RankFormer, published member | -0.00166 | [-0.00194, -0.00139] | 0.001 |
| D_ens − PE-RankFormer, published member | -0.00008 | [-0.00031, +0.00014] | 0.501 |
| E_ens − PE-RankFormer, published member | +0.00008 | [-0.00015, +0.00031] | 0.482 |
| arm F, 3-seed ensemble (canonical batching + features) − PE-RankFormer, published member | -0.00024 | [-0.00047, -0.00001] | 0.048 |
| H_ens − PE-RankFormer, published member | +0.00029 | [+0.00005, +0.00053] | 0.013 |
| arm F, 3-seed ensemble (canonical batching + features) − arm A, 3-seed ensemble (released recipe) | +0.00142 | [+0.00117, +0.00168] | 0.001 |
| arm F, 3-seed ensemble (canonical batching + features) − E_ens | -0.00031 | [-0.00054, -0.00007] | 0.012 |
| H_ens − E_ens | +0.00022 | [-0.00002, +0.00045] | 0.07 |
| E_ens − arm A, 3-seed ensemble (released recipe) | +0.00174 | [+0.00146, +0.00202] | 0.001 |
| PE-RankFormer, published member − OptiPrime | -0.00078 | [-0.00111, -0.00046] | 0.001 |

## depth 8+ — 2,400 groups, 2,371 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0293 | 0.1130 | 0.06443 |
| OptiPrime | 0.0645 | 0.3617 | 0.02922 |
| PE-RankFormer, published member | 0.0620 | 0.3417 | 0.03171 |
| arm A, 3-seed ensemble (released recipe) | 0.0573 | 0.2954 | 0.03639 |
| arm F, 3-seed ensemble (canonical batching + features) | 0.0618 | 0.3279 | 0.03190 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| arm A, 3-seed ensemble (released recipe) − OptiPrime | -0.00717 | [-0.00897, -0.00546] | 0.001 |
| D_ens − OptiPrime | -0.00285 | [-0.00448, -0.00129] | 0.001 |
| E_ens − OptiPrime | -0.00218 | [-0.00382, -0.00058] | 0.01 |
| arm F, 3-seed ensemble (canonical batching + features) − OptiPrime | -0.00268 | [-0.00436, -0.00107] | 0.001 |
| H_ens − OptiPrime | -0.00099 | [-0.00261, +0.00060] | 0.213 |
| arm A, 3-seed ensemble (released recipe) − PE-RankFormer, published member | -0.00468 | [-0.00616, -0.00318] | 0.001 |
| D_ens − PE-RankFormer, published member | -0.00037 | [-0.00156, +0.00079] | 0.531 |
| E_ens − PE-RankFormer, published member | +0.00030 | [-0.00094, +0.00142] | 0.622 |
| arm F, 3-seed ensemble (canonical batching + features) − PE-RankFormer, published member | -0.00020 | [-0.00152, +0.00106] | 0.778 |
| H_ens − PE-RankFormer, published member | +0.00150 | [+0.00023, +0.00268] | 0.018 |
| arm F, 3-seed ensemble (canonical batching + features) − arm A, 3-seed ensemble (released recipe) | +0.00449 | [+0.00308, +0.00598] | 0.001 |
| arm F, 3-seed ensemble (canonical batching + features) − E_ens | -0.00050 | [-0.00176, +0.00073] | 0.44 |
| H_ens − E_ens | +0.00119 | [+0.00004, +0.00237] | 0.047 |
| E_ens − arm A, 3-seed ensemble (released recipe) | +0.00499 | [+0.00353, +0.00644] | 0.001 |
| PE-RankFormer, published member − OptiPrime | -0.00249 | [-0.00416, -0.00083] | 0.004 |

## decision worth >= 0.05 — 5,614 groups, 5,286 sites

| model | achieved @1 | hit rate @1 | regret @1 |
|---|---:|---:|---:|
| random choice | 0.0561 | 0.2197 | 0.08220 |
| OptiPrime | 0.1099 | 0.5891 | 0.02842 |
| PE-RankFormer, published member | 0.1066 | 0.5549 | 0.03171 |
| arm A, 3-seed ensemble (released recipe) | 0.1015 | 0.5110 | 0.03680 |
| arm F, 3-seed ensemble (canonical batching + features) | 0.1060 | 0.5435 | 0.03235 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| arm A, 3-seed ensemble (released recipe) − OptiPrime | -0.00838 | [-0.00975, -0.00704] | 0.001 |
| D_ens − OptiPrime | -0.00307 | [-0.00425, -0.00185] | 0.001 |
| E_ens − OptiPrime | -0.00294 | [-0.00414, -0.00175] | 0.001 |
| arm F, 3-seed ensemble (canonical batching + features) − OptiPrime | -0.00394 | [-0.00513, -0.00271] | 0.001 |
| H_ens − OptiPrime | -0.00195 | [-0.00309, -0.00074] | 0.001 |
| arm A, 3-seed ensemble (released recipe) − PE-RankFormer, published member | -0.00509 | [-0.00615, -0.00404] | 0.001 |
| D_ens − PE-RankFormer, published member | +0.00021 | [-0.00068, +0.00106] | 0.639 |
| E_ens − PE-RankFormer, published member | +0.00035 | [-0.00055, +0.00121] | 0.418 |
| arm F, 3-seed ensemble (canonical batching + features) − PE-RankFormer, published member | -0.00065 | [-0.00158, +0.00029] | 0.201 |
| H_ens − PE-RankFormer, published member | +0.00133 | [+0.00043, +0.00220] | 0.004 |
| arm F, 3-seed ensemble (canonical batching + features) − arm A, 3-seed ensemble (released recipe) | +0.00445 | [+0.00334, +0.00555] | 0.001 |
| arm F, 3-seed ensemble (canonical batching + features) − E_ens | -0.00100 | [-0.00192, -0.00007] | 0.041 |
| H_ens − E_ens | +0.00098 | [+0.00005, +0.00189] | 0.032 |
| E_ens − arm A, 3-seed ensemble (released recipe) | +0.00544 | [+0.00438, +0.00653] | 0.001 |
| PE-RankFormer, published member − OptiPrime | -0.00329 | [-0.00451, -0.00209] | 0.001 |

Pooled Spearman on the panel: OptiPrime 0.6931, PE-RankFormer, published member 0.7091, arm A, 3-seed ensemble (released recipe) 0.6908, D_ens 0.7019, E_ens 0.7070, arm F, 3-seed ensemble (canonical batching + features) 0.7071, H_ens 0.7113.
