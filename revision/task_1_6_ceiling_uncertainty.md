# Task 1.6 — uncertainty on the noise ceiling

Ceiling estimated from **649 replicate groups**
(1298 ordered pairs after symmetrisation). Interval from resampling
the groups with replacement, 400 resamples, seed 20260903. Commit
`d47d183082cb`.

| surface | Kim rows | model ρ | ceiling | 95% CI on ceiling | gap | 95% CI on gap |
|---|---:|---:|---:|---|---:|---|
| dev_fold_0 | 19,747 | 0.7869 | 0.9022 | [0.8424, 0.9257] | +0.1152 | [+0.0554, +0.1388] |
| dev_fold_1 | 19,740 | 0.7883 | 0.9083 | [0.8526, 0.9259] | +0.1200 | [+0.0643, +0.1376] |
| dev_fold_2 | 19,742 | 0.7985 | 0.9071 | [0.8492, 0.9278] | +0.1086 | [+0.0507, +0.1294] |
| heldout | 11,334 | 0.8124 | 0.9164 | [0.8653, 0.9327] | +0.1040 | [+0.0529, +0.1204] |

Estimator noise alone, holding the group set fixed, has SD: dev_fold_0 0.0003, dev_fold_1 0.0003, dev_fold_2 0.0002, heldout 0.0003 — an order of
magnitude smaller than the group-resampling interval, so the uncertainty is dominated by
having only 649 replicate groups, not by the synthetic draw.

The headroom claim survives the interval on every surface: the lower bound of the gap
stays well clear of zero. What the interval does change is the precision with which the
headroom can be quoted — "roughly +0.10" is supportable, a third decimal place is not.
