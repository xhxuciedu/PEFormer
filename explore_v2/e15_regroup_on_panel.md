# E15 - the regrouping repair, on the panel where the deficit was found

> **Superseded in part by E16.** Same four endpoint defects as E11, including the random hit rate of 0.3354 shown below, which is correctly 0.4442. Corrected values: canonical − control +0.00184, canonical − OptiPrime −0.00126, both p = 0.001. See `e16_corrected_endpoints.md`.

> **Disclosure.** Second use of the reserved panel. Training committed 13:22:59; E11's panel head-to-head written 14:43:22; no configuration changed after 13:22:59; epoch selection used fold 1 only. This is a secondary comparison, not a second confirmation; the next confirmatory claim needs a sealed surface.

## Pooled Spearman on the panel

| model | pooled rho |
|---|---:|
| PE-RankFormer, published member | 0.7091 |
| OptiPrime | 0.6931 |
| retrained, current ranking key | 0.6822 |
| retrained, canonical decision-group key | 0.6968 |

## all eligible groups (depth >=2)

30,475 groups over 22,931 target sites; oracle 0.0461.

| model | achieved @1 | @3 | best-design hit rate | regret @1 |
|---|---:|---:|---:|---:|
| random choice | 0.0217 | 0.0469 | 0.3354 | 0.02443 |
| OptiPrime | 0.0369 | 0.0558 | 0.6647 | 0.00922 |
| PE-RankFormer, published member | 0.0363 | 0.0557 | 0.6599 | 0.00985 |
| retrained, current ranking key | 0.0340 | 0.0548 | 0.6300 | 0.01217 |
| retrained, canonical decision-group key | 0.0357 | 0.0555 | 0.6471 | 0.01039 |

| paired difference, achieved efficiency @1 | value | 95% CI | p |
|---|---:|---|---:|
| retrained, canonical decision-group key − OptiPrime | -0.00117 | [-0.00143, -0.00091] | 0.0005 |
| retrained, current ranking key − OptiPrime | -0.00295 | [-0.00325, -0.00266] | 0.0005 |
| retrained, canonical decision-group key − retrained, current ranking key | +0.00178 | [+0.00153, +0.00204] | 0.0005 |
| PE-RankFormer, published member − OptiPrime | -0.00062 | [-0.00088, -0.00037] | 0.0005 |

## depth 5-7

6,375 groups over 6,205 target sites; oracle 0.0675.

| model | achieved @1 | @3 | best-design hit rate | regret @1 |
|---|---:|---:|---:|---:|
| random choice | 0.0275 | 0.0525 | 0.1819 | 0.03993 |
| OptiPrime | 0.0520 | 0.0651 | 0.5071 | 0.01546 |
| PE-RankFormer, published member | 0.0509 | 0.0651 | 0.4916 | 0.01656 |
| retrained, current ranking key | 0.0467 | 0.0640 | 0.4417 | 0.02075 |
| retrained, canonical decision-group key | 0.0499 | 0.0649 | 0.4747 | 0.01751 |

| paired difference, achieved efficiency @1 | value | 95% CI | p |
|---|---:|---|---:|
| retrained, canonical decision-group key − OptiPrime | -0.00205 | [-0.00280, -0.00128] | 0.0005 |
| retrained, current ranking key − OptiPrime | -0.00529 | [-0.00611, -0.00445] | 0.0005 |
| retrained, canonical decision-group key − retrained, current ranking key | +0.00324 | [+0.00247, +0.00400] | 0.0005 |
| PE-RankFormer, published member − OptiPrime | -0.00110 | [-0.00181, -0.00037] | 0.001 |

## depth 8+

2,400 groups over 2,371 target sites; oracle 0.0937.

| model | achieved @1 | @3 | best-design hit rate | regret @1 |
|---|---:|---:|---:|---:|
| random choice | 0.0293 | 0.0587 | 0.1023 | 0.06443 |
| OptiPrime | 0.0645 | 0.0845 | 0.3617 | 0.02922 |
| PE-RankFormer, published member | 0.0620 | 0.0842 | 0.3429 | 0.03168 |
| retrained, current ranking key | 0.0559 | 0.0804 | 0.2888 | 0.03787 |
| retrained, canonical decision-group key | 0.0611 | 0.0830 | 0.3296 | 0.03266 |

| paired difference, achieved efficiency @1 | value | 95% CI | p |
|---|---:|---|---:|
| retrained, canonical decision-group key − OptiPrime | -0.00344 | [-0.00509, -0.00174] | 0.0005 |
| retrained, current ranking key − OptiPrime | -0.00865 | [-0.01051, -0.00691] | 0.0005 |
| retrained, canonical decision-group key − retrained, current ranking key | +0.00521 | [+0.00356, +0.00685] | 0.0005 |
| PE-RankFormer, published member − OptiPrime | -0.00246 | [-0.00413, -0.00082] | 0.006 |

## decision worth >= 0.05

5,615 groups over 5,287 target sites; oracle 0.1383.

| model | achieved @1 | @3 | best-design hit rate | regret @1 |
|---|---:|---:|---:|---:|
| random choice | 0.0561 | 0.1061 | 0.2198 | 0.08219 |
| OptiPrime | 0.1099 | 0.1313 | 0.5891 | 0.02841 |
| PE-RankFormer, published member | 0.1066 | 0.1310 | 0.5553 | 0.03169 |
| retrained, current ranking key | 0.0978 | 0.1282 | 0.4835 | 0.04048 |
| retrained, canonical decision-group key | 0.1049 | 0.1303 | 0.5361 | 0.03341 |

| paired difference, achieved efficiency @1 | value | 95% CI | p |
|---|---:|---|---:|
| retrained, canonical decision-group key − OptiPrime | -0.00500 | [-0.00623, -0.00375] | 0.0005 |
| retrained, current ranking key − OptiPrime | -0.01207 | [-0.01340, -0.01058] | 0.0005 |
| retrained, canonical decision-group key − retrained, current ranking key | +0.00707 | [+0.00580, +0.00822] | 0.0005 |
| PE-RankFormer, published member − OptiPrime | -0.00328 | [-0.00451, -0.00203] | 0.0005 |