# E11 - the pre-declared comparison on the reserved panel

> **Superseded in part by E16.** This report's endpoints were computed by inline code with four defects: a candidate was a row rather than a distinct design, @k was computed wherever rows allowed, the random hit rate ignored tied maximisers (0.3354 here against a correct 0.4442), and the bootstrap returned its floor p-value for identical arms. `e16_corrected_endpoints.md` recomputes everything through the tested `endpoints` module. The model-versus-model conclusions are unchanged; ours − OptiPrime becomes −0.00064 (p = 0.001).

118,187 candidate measurements scored. Canonicaliser version 2. OptiPrime: joined, coverage 1.0000.

Pooled Spearman on the panel, as a reconstruction sanity check (not an endpoint): ours 0.7091, op 0.6931.

Exposure: PE-RankFormer's ordinal-S4D backbone and OptiPrime were both trained on Kim's LibSmall files and **neither** on this panel's source library. Released DeepPrime was trained on it and is therefore excluded as a comparator here.

Reading the table: a group whose designs tie at the maximum gives every predictor a hit, which is why random selection's best-design hit rate is well above 1/depth -- 5,807 groups have all designs at exactly zero. The pre-registration requires tied optima to receive full credit and an all-zero tie not to be presented as success, so the all-eligible stratum is the deployment number and the informative stratum is the discrimination number, and both are reported.


## all eligible groups (depth >=2)

30,475 groups over 22,931 target sites; 5,807 have every design at exactly zero; oracle achieves 0.0461. Groups eligible at each budget: {'1': 30475, '3': 18660, '5': 8775}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0217 | 0.0469 | 0.0673 | 0.3354 | 0.02443 |
| OptiPrime | 0.0369 | 0.0558 | 0.0737 | 0.6647 | 0.00922 |
| PE-RankFormer ordinal-S4D | 0.0363 | 0.0557 | 0.0738 | 0.6599 | 0.00985 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00062 | [-0.00088, -0.00037] | 0.0005 |
| achieved efficiency at 3 | -0.00003 | [-0.00019, +0.00013] | 0.674 |
| best design hit rate | -0.00482 | [-0.00968, -0.00007] | 0.044 |
| regret at 1 | +0.00062 | [+0.00037, +0.00088] | 0.0005 |

## informative groups (non-constant outcome)

24,668 groups over 19,642 target sites; 0 have every design at exactly zero; oracle achieves 0.0570. Groups eligible at each budget: {'1': 24668, '3': 16773, '5': 8421}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0268 | 0.0521 | 0.0702 | 0.3133 | 0.03018 |
| OptiPrime | 0.0456 | 0.0620 | 0.0769 | 0.5857 | 0.01139 |
| PE-RankFormer ordinal-S4D | 0.0448 | 0.0620 | 0.0769 | 0.5798 | 0.01216 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00077 | [-0.00110, -0.00044] | 0.0005 |
| achieved efficiency at 3 | -0.00004 | [-0.00022, +0.00014] | 0.664 |
| best design hit rate | -0.00596 | [-0.01218, -0.00008] | 0.048 |
| regret at 1 | +0.00077 | [+0.00044, +0.00110] | 0.0005 |

## depth 2

11,815 groups over 9,457 target sites; 3,920 have every design at exactly zero; oracle achieves 0.0275. Groups eligible at each budget: {'1': 11815, '3': 0, '5': 0}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0176 | 0.0026 | - | 0.5000 | 0.00994 |
| OptiPrime | 0.0243 | 0.0026 | - | 0.8162 | 0.00326 |
| PE-RankFormer ordinal-S4D | 0.0240 | 0.0026 | - | 0.8177 | 0.00349 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00023 | [-0.00048, +0.00003] | 0.082 |
| achieved efficiency at 3 | +0.00000 | [+0.00000, +0.00000] | 0.0005 |
| best design hit rate | +0.00152 | [-0.00484, +0.00792] | 0.648 |
| regret at 1 | +0.00023 | [-0.00003, +0.00048] | 0.082 |

## depth 3-4

9,885 groups over 9,337 target sites; 1,533 have every design at exactly zero; oracle achieves 0.0431. Groups eligible at each budget: {'1': 9885, '3': 9885, '5': 0}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0210 | 0.0404 | 0.0000 | 0.2944 | 0.02204 |
| OptiPrime | 0.0356 | 0.0428 | 0.0000 | 0.6588 | 0.00748 |
| PE-RankFormer ordinal-S4D | 0.0352 | 0.0428 | 0.0000 | 0.6567 | 0.00782 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00034 | [-0.00074, +0.00005] | 0.093 |
| achieved efficiency at 3 | -0.00001 | [-0.00008, +0.00006] | 0.886 |
| best design hit rate | -0.00212 | [-0.01102, +0.00628] | 0.616 |
| regret at 1 | +0.00034 | [-0.00005, +0.00074] | 0.093 |

## depth 5-7

6,375 groups over 6,205 target sites; 325 have every design at exactly zero; oracle achieves 0.0675. Groups eligible at each budget: {'1': 6375, '3': 6375, '5': 6375}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0275 | 0.0525 | 0.0648 | 0.1819 | 0.03993 |
| OptiPrime | 0.0520 | 0.0651 | 0.0673 | 0.5071 | 0.01546 |
| PE-RankFormer ordinal-S4D | 0.0509 | 0.0651 | 0.0673 | 0.4916 | 0.01656 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00110 | [-0.00181, -0.00037] | 0.001 |
| achieved efficiency at 3 | +0.00004 | [-0.00026, +0.00034] | 0.782 |
| best design hit rate | -0.01553 | [-0.02751, -0.00299] | 0.015 |
| regret at 1 | +0.00110 | [+0.00037, +0.00181] | 0.001 |

## depth 8+

2,400 groups over 2,371 target sites; 29 have every design at exactly zero; oracle achieves 0.0937. Groups eligible at each budget: {'1': 2400, '3': 2400, '5': 2400}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0293 | 0.0587 | 0.0741 | 0.1023 | 0.06443 |
| OptiPrime | 0.0645 | 0.0845 | 0.0908 | 0.3617 | 0.02922 |
| PE-RankFormer ordinal-S4D | 0.0620 | 0.0842 | 0.0911 | 0.3429 | 0.03168 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00246 | [-0.00413, -0.00082] | 0.006 |
| achieved efficiency at 3 | -0.00033 | [-0.00127, +0.00060] | 0.461 |
| best design hit rate | -0.01875 | [-0.04012, +0.00168] | 0.07 |
| regret at 1 | +0.00246 | [+0.00082, +0.00413] | 0.006 |

## decision worth < 0.005

12,553 groups over 10,312 target sites; 5,807 have every design at exactly zero; oracle achieves 0.0034. Groups eligible at each budget: {'1': 12553, '3': 5039, '5': 1323}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0023 | 0.0022 | 0.0024 | 0.4073 | 0.00102 |
| OptiPrime | 0.0027 | 0.0024 | 0.0025 | 0.7520 | 0.00065 |
| PE-RankFormer ordinal-S4D | 0.0027 | 0.0024 | 0.0025 | 0.7543 | 0.00064 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | +0.00001 | [-0.00002, +0.00003] | 0.525 |
| achieved efficiency at 3 | +0.00001 | [-0.00000, +0.00002] | 0.084 |
| best design hit rate | +0.00231 | [-0.00369, +0.00804] | 0.464 |
| regret at 1 | -0.00001 | [-0.00003, +0.00002] | 0.525 |

## decision worth 0.005-0.02

6,346 groups over 5,942 target sites; 0 have every design at exactly zero; oracle achieves 0.0279. Groups eligible at each budget: {'1': 6346, '3': 4057, '5': 1595}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0165 | 0.0198 | 0.0181 | 0.3369 | 0.01143 |
| OptiPrime | 0.0226 | 0.0217 | 0.0191 | 0.6193 | 0.00526 |
| PE-RankFormer ordinal-S4D | 0.0227 | 0.0218 | 0.0191 | 0.6239 | 0.00519 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | +0.00007 | [-0.00013, +0.00028] | 0.521 |
| achieved efficiency at 3 | +0.00004 | [-0.00003, +0.00012] | 0.212 |
| best design hit rate | +0.00457 | [-0.00707, +0.01564] | 0.46 |
| regret at 1 | -0.00007 | [-0.00028, +0.00013] | 0.521 |

## decision worth 0.02-0.05

5,961 groups over 5,634 target sites; 0 have every design at exactly zero; oracle achieves 0.0688. Groups eligible at each budget: {'1': 5961, '3': 4537, '5': 2304}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0356 | 0.0550 | 0.0545 | 0.2915 | 0.03315 |
| OptiPrime | 0.0554 | 0.0618 | 0.0578 | 0.6002 | 0.01341 |
| PE-RankFormer ordinal-S4D | 0.0552 | 0.0619 | 0.0578 | 0.5977 | 0.01360 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00019 | [-0.00074, +0.00038] | 0.534 |
| achieved efficiency at 3 | +0.00019 | [-0.00001, +0.00039] | 0.06 |
| best design hit rate | -0.00252 | [-0.01491, +0.01041] | 0.696 |
| regret at 1 | +0.00019 | [-0.00038, +0.00074] | 0.534 |

## decision worth >= 0.05

5,615 groups over 5,287 target sites; 0 have every design at exactly zero; oracle achieves 0.1383. Groups eligible at each budget: {'1': 5615, '3': 5027, '5': 3553}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0561 | 0.1061 | 0.1220 | 0.2198 | 0.08219 |
| OptiPrime | 0.1099 | 0.1313 | 0.1352 | 0.5891 | 0.02841 |
| PE-RankFormer ordinal-S4D | 0.1066 | 0.1310 | 0.1353 | 0.5553 | 0.03169 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00328 | [-0.00451, -0.00203] | 0.0005 |
| achieved efficiency at 3 | -0.00034 | [-0.00090, +0.00019] | 0.203 |
| best design hit rate | -0.03384 | [-0.04701, -0.02093] | 0.0005 |
| regret at 1 | +0.00328 | [+0.00203, +0.00451] | 0.0005 |

## edit type sub

15,341 groups over 12,712 target sites; 1,982 have every design at exactly zero; oracle achieves 0.0497. Groups eligible at each budget: {'1': 15341, '3': 9778, '5': 4409}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0241 | 0.0488 | 0.0692 | 0.3321 | 0.02566 |
| OptiPrime | 0.0405 | 0.0570 | 0.0744 | 0.6459 | 0.00929 |
| PE-RankFormer ordinal-S4D | 0.0398 | 0.0571 | 0.0743 | 0.6385 | 0.00991 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00062 | [-0.00099, -0.00024] | 0.0005 |
| achieved efficiency at 3 | +0.00004 | [-0.00016, +0.00024] | 0.698 |
| best design hit rate | -0.00743 | [-0.01467, -0.00065] | 0.037 |
| regret at 1 | +0.00062 | [+0.00024, +0.00099] | 0.0005 |

## edit type ins

8,909 groups over 6,463 target sites; 2,549 have every design at exactly zero; oracle achieves 0.0425. Groups eligible at each budget: {'1': 8909, '3': 5019, '5': 2466}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0193 | 0.0450 | 0.0658 | 0.3450 | 0.02319 |
| OptiPrime | 0.0336 | 0.0551 | 0.0740 | 0.7146 | 0.00893 |
| PE-RankFormer ordinal-S4D | 0.0328 | 0.0549 | 0.0741 | 0.7072 | 0.00973 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00079 | [-0.00127, -0.00032] | 0.002 |
| achieved efficiency at 3 | -0.00022 | [-0.00059, +0.00016] | 0.265 |
| best design hit rate | -0.00741 | [-0.01645, +0.00113] | 0.089 |
| regret at 1 | +0.00079 | [+0.00032, +0.00127] | 0.002 |

## edit type del

6,225 groups over 4,871 target sites; 1,276 have every design at exactly zero; oracle achieves 0.0424. Groups eligible at each budget: {'1': 6225, '3': 3863, '5': 1900}.

| predictor | achieved @1 | achieved @3 | achieved @5 | best-design hit rate | regret @1 |
|---|---|---|---|---|---|
| random choice | 0.0192 | 0.0443 | 0.0650 | 0.3299 | 0.02317 |
| OptiPrime | 0.0329 | 0.0534 | 0.0720 | 0.6395 | 0.00948 |
| PE-RankFormer ordinal-S4D | 0.0326 | 0.0534 | 0.0721 | 0.6448 | 0.00986 |

| paired difference (ours - OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency at 1 | -0.00038 | [-0.00093, +0.00017] | 0.164 |
| achieved efficiency at 3 | +0.00001 | [-0.00034, +0.00036] | 0.963 |
| best design hit rate | +0.00530 | [-0.00605, +0.01573] | 0.332 |
| regret at 1 | +0.00038 | [-0.00017, +0.00093] | 0.164 |

## Screening effort, fixed population (depth >= 8)

2,400 decision groups, every budget scored on the same groups. Mean achieved efficiency of the best of k nominated candidates.

| k | PE-RankFormer | OptiPrime | random |
|---|---|---|---|
| 1 | 0.0620 | 0.0645 | 0.0293 |
| 2 | 0.0764 | 0.0783 | 0.0469 |
| 3 | 0.0842 | 0.0845 | 0.0587 |
| 4 | 0.0885 | 0.0885 | 0.0674 |
| 5 | 0.0911 | 0.0908 | 0.0741 |
| 6 | 0.0924 | 0.0920 | 0.0794 |
| 7 | 0.0930 | 0.0928 | 0.0838 |
| 8 | 0.0934 | 0.0933 | 0.0876 |

**Random selection needs 4 candidates to match what PE-RankFormer's single nominated candidate achieves.**
 OptiPrime needs 1.


Dependence unit is the target site, resampled with replacement; 22,931 sites in the primary stratum.
