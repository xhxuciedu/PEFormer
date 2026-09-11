# E21 - the headline correlation metrics, on both surfaces

The manuscript's headline is a pooled rank correlation, with Pearson on the isotonic-calibrated scale because the ordinal head emits rank estimates rather than efficiencies. **Both headline claims replicate on the reserved panel**, a library neither model was trained on.

## held-out fold 0 — 20,509 rows, 750 clusters

| model | Spearman | Pearson |
|---|---:|---:|
| OptiPrime | 0.8690 | 0.8270 |
| PE-RankFormer, raw score | 0.9079 | 0.7278 |
| PE-RankFormer, calibrated | 0.9079 | 0.8637 |

| margin over OptiPrime | value | 95% CI | p |
|---|---:|---|---:|
| Spearman (raw score) | +0.0389 | [+0.0290, +0.0495] | 0.0005 |
| Pearson (calibrated) | +0.0366 | [+0.0203, +0.0557] | 0.0005 |

## reserved panel — 118,187 rows, 30,302 clusters

| model | Spearman | Pearson |
|---|---:|---:|
| OptiPrime | 0.6931 | 0.5854 |
| PE-RankFormer, raw score | 0.7091 | 0.5522 |
| PE-RankFormer, calibrated | 0.7089 | 0.6017 |

| margin over OptiPrime | value | 95% CI | p |
|---|---:|---|---:|
| Spearman (raw score) | +0.0159 | [+0.0136, +0.0182] | 0.0005 |
| Pearson (calibrated) | +0.0163 | [+0.0119, +0.0207] | 0.0005 |

## The frozen calibrator transfers to an unseen library

Applied unchanged to the panel, the development-fitted isotonic map lifts Pearson from 0.5522 to **0.6017** (+0.0495) while leaving the rank correlation at 0.7089. A monotone map cannot change a rank correlation; the third-decimal Spearman difference is out-of-range clipping creating ties at the boundaries. The map was fitted on development out-of-fold predictions and is applied here unchanged to a library it never saw.

## Reading this beside the decision result

These are not in conflict; they are different estimands measured on the same rows. Pooled correlation asks how well the whole list is ordered, and mixes which locus is easy with which design is best; PE-RankFormer is ahead on it on both surfaces. The fixed-allele decision asks which of the interchangeable designs for one intended allele to order, and on the panel OptiPrime is ahead on that (E16). The manuscript is entitled to the correlation claim, on two surfaces now rather than one, and is not entitled to a deployment-superiority claim.
