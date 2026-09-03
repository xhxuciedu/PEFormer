# Task 1.5 — calibration against a floor

Protospacer-clustered bootstrap, 5000 resamples, seed 20260903.
Commit `d47d183082cb`. Constant baselines are fitted on the
297,962 training rows (median 0.0856, mean 0.2211); the held-out
mean 0.1133 is shown as an oracle floor no honest model could use.

| predictor | MAE | RMSE | bias | 95% CI on MAE |
|---|---:|---:|---:|---|
| constant at training median | 0.1227 | 0.1827 | -0.0276 | [0.1144, 0.1313] |
| constant at training mean | 0.1892 | 0.2103 | +0.1078 | [0.1851, 0.1937] |
| constant at held-out mean (oracle floor) | 0.1328 | 0.1806 | +0.0000 | [0.1263, 0.1398] |
| OptiPrime | 0.0590 | 0.1026 | +0.0127 | [0.0549, 0.0629] |
| PE-RankFormer + isotonic | 0.0478 | 0.0912 | -0.0038 | [0.0431, 0.0523] |

MAE against OptiPrime: -0.0112
(95% CI [-0.0142, -0.0084], p = 0.0002); negative
favours PE-RankFormer.

## Per-decile calibration, by predicted rank

| decile | n | mean predicted | mean observed | gap (ours) | gap (OptiPrime) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,051 | 0.0001 | 0.0002 | -0.0000 | +0.0010 |
| 2 | 2,051 | 0.0007 | 0.0005 | +0.0002 | +0.0044 |
| 3 | 2,051 | 0.0019 | 0.0021 | -0.0001 | +0.0097 |
| 4 | 2,051 | 0.0071 | 0.0098 | -0.0027 | +0.0185 |
| 5 | 2,051 | 0.0229 | 0.0251 | -0.0021 | +0.0309 |
| 6 | 2,050 | 0.0492 | 0.0592 | -0.0100 | +0.0311 |
| 7 | 2,051 | 0.0923 | 0.1041 | -0.0118 | +0.0336 |
| 8 | 2,051 | 0.1574 | 0.1666 | -0.0092 | +0.0288 |
| 9 | 2,051 | 0.2612 | 0.2744 | -0.0132 | +0.0082 |
| 10 | 2,051 | 0.5013 | 0.4907 | +0.0106 | -0.0392 |

## The high-efficiency tail, where absolute predictions matter for design

| slice | n | mean predicted | mean observed | gap | MAE | OptiPrime gap |
|---|---:|---:|---:|---:|---:|---:|
| model's top 5% | 1,025 | 0.5987 | 0.5730 | +0.0257 | 0.1265 | -0.0524 |
| model's top 1% | 205 | 0.7985 | 0.7334 | +0.0651 | 0.1198 | -0.1130 |
