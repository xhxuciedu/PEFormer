# Phase 0 — consistency audit

Produced by `revision/audit_00_consistency.py` (machine-readable:
`revision/00_consistency_audit.json`), which records the git commit and input file
hashes and takes `--seed`. Every value below is read back from a data file or a
checkpoint, not from the manuscript.

Manuscript audited: `reports/paper/pe_rankformer_paper.tex`.

## Summary

All four named suspects are **resolved**, and none was a wrong number: in every case
two or more *correct* figures referring to *different populations* were presented
without saying which. One (the threshold count) additionally had the wrong value for
the reported system. Three of the four were already corrected in this branch before
this audit ran; the fourth is corrected now.

| # | Suspect | Verdict | Status |
|---|---|---|---|
| 1 | zero-mass 16.0% vs 28.4% | both correct, different populations | fixed earlier this branch |
| 2 | three Kim (n, ρ) triples | all three correct, different model × surface | fixed earlier this branch |
| 3 | Kim zero fraction 49.9% vs 50.7% | both correct, different populations | fixed earlier this branch |
| 4 | "K=20 yields 17–18"; Fig 1C says 17 | **manuscript was wrong for the reported system** | fixed by this audit |

## Suspect 1 and 3 — zero-mass populations

Measured on `optiprime_official_318471.parquet` and `round3_dev_assignments.parquet`:

| population | n | zero-mass | Kim zero-mass |
|---|---:|---:|---:|
| Training rows (`fold∈1..5`) | 297,962 | **15.96% → 16.0%** | **49.87% → 49.9%** |
| Held-out (`fold==0`) | 20,509 | **26.82% → 26.8%** | 47.70% |
| Dev fold 0, `val` rows | 35,649 | **28.44% → 28.4%** | 50.81% |
| Dev fold 1, `val` rows | 35,692 | 28.27% | 50.59% |
| Dev fold 2, `val` rows | 35,700 | 28.41% | 50.83% |
| Mean of the three dev `val` folds | 35,680 | 28.38% | **50.74% → 50.7%** |

So: **16.0% is the training corpus, 28.4% is development fold 0, 26.8% is held-out;
49.9% is Kim within training, 50.7% is Kim within the development folds.** All five
figures are individually right. The manuscript used 28.4% to motivate a *training*
objective, where 16.0% is the relevant number, and quoted 49.9% and 50.7% two lines
apart in the tie section. Both now labelled at point of use. Fig 1C's "16.0% of training
rows are exactly zero" is correct as printed.

## Suspect 2 — the three Kim numbers

| manuscript appearance | what it actually is | n | ρ |
|---|---|---:|---:|
| Table 3, Kim column | frozen 5-member rank-average ensemble, **held-out** | 11,334 | **0.8124** |
| Table 6, ordinal+S4D | that single member alone, **held-out** | 11,334 | **0.8151** |
| Table 8 (old), "all Kim rows" | ordinal+S4D out-of-fold, **development fold 0 only** | 19,747 | **0.7869** |
| — | same model, mean over all three dev folds | 19,743 | 0.7912 |

Per-fold: 0.7869 (f0), 0.7883 (f1), 0.7985 (f2). All four numbers are correct. The old
Table 8 labelled a single development fold "all Kim rows" and set it against a ceiling,
while Table 3's Kim figure is a different model on a different surface — a reader
comparing the two was comparing nothing. Table 8 now reports all three dev folds and
both held-out variants explicitly (`scripts/evaluate/noise_ceiling_surfaces.py`).

## Suspect 4 — ordinal threshold count

Read back from the `model_config` of all 156 checkpoints carrying `best.pt`
(98 have an ordinal head). Note `model_config` is a **dict**, so `getattr` silently
returns `None` — an earlier pass of this audit reported "no thresholds" for every
checkpoint because of exactly that.

| K−1 | checkpoints | which runs |
|---:|---:|---|
| **17** | **20** | **`r4p2_{ordSSM,ordA,ordB,ordC}_cv{1..5}` — every official-fold checkpoint of the reported system** |
| 18 | 69 | matched development-fold runs |
| 19 | 6 | round-7 ctx-primary (within-condition quantiles) |
| 16 | 1 | `r8_liukim` (trained on two sources, fewer distinct quantiles) |
| 7 | 1 | the K=8 ablation |
| 43 | 1 | the K=50 ablation |

**The reported system has exactly 17 thresholds, in all twenty of its checkpoints.**
"17–18" is true of the study as a whole but not of the model the paper reports, and
Fig 1C's 17 was right. The difference is that official folds train on 4/5 of 297,962
rows while dev folds train on a different subset, leaving one fewer duplicate quantile
at the bottom of the distribution. Methods, the hyperparameter table and the §Methods
note are corrected, and now say where the count was read from.

## Duplicated-quantity sweep

61 four-decimal or percentage quantities appear more than once. Every one was checked
for population/surface agreement; **no further disagreements found.** Values appearing
four or more times are all legitimate repeats of a single quantity across abstract,
contributions, table and prose.

Two additional **value collisions** — distinct quantities that happen to share a
printed value — are worth knowing about because each invites the suspicion of a
copy-paste error:

| value | the distinct quantities sharing it |
|---|---|
| `0.9082` | ordinal+S4D **development** OOF (Table 4, §factorial prose) and ordinal+S4D **held-out** (Table 6, §ensemble prose). Verified distinct at the 5th decimal on disjoint rows (0.908179 vs 0.908155); already footnoted |
| `0.0031` | (a) max cost of patience-5 early stopping, (b) ordinal+S4D marginal ensemble contribution, (c) selective SSM vs its frozen control. Three unrelated quantities |

The `0.9082` collision carries a footnote. I have not added one for `0.0031` — a second
such footnote would cost more in noise than it buys, and the three appearances are in
sections far apart. Flagging it here so it is a known, not a latent, hazard.

## Figures regenerated

Both figures the brief flagged had rendering defects; neither was the panel named.

**Figure 5C** was the source of `0.908.908`, and was worse than a label collision:
`ylim` was set to 0.62 while the Spearman and Pearson bars reach 0.86–0.91, so those
bars **overflowed the axes and printed their value labels outside the panel**, floating
above it. Two of the four labels then overlapped. The panel also compared the calibrated
output against our own *uncalibrated* rank average, whose MAE and RMSE are not in
efficiency units, and its title claimed the "8-fold" reduction this branch removed from
the text. Rebuilt: three series (OptiPrime, ours rank-average, ours + calibration) with
`n/a` marked where a quantity has no interpretation, `ylim` 1.45, capped y-ticks,
rotated x-labels, a two-column legend in the headroom band, and a title matching the
revised claim.

**Figure 1, panel B** (not 1E — 1E renders correctly) had the overlapping text: the
mixer captions were three lines at 6.4 pt inside boxes whose rendered height was almost
exactly the text height, so the first and last lines sat outside the box edges, and the
four arrows landed at x=2.5 and x=7.5 which is *inside* the caption text. Rebuilt with
two-line captions in taller boxes and a single arrow down the 0.4-unit gap between the
two alternative boxes — which is also the semantically correct place, the boxes being
alternatives.

## Unresolved

Nothing. No quantity in the manuscript is left unreconciled by this audit.
