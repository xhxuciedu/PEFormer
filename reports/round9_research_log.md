# Round 9 — submission audit, a comparison-fairness finding, and two new mechanisms

Round 9 began as a pre-submission audit of `reports/paper/pe_rankformer_paper.tex`
rather than as a model-search round. The audit turned up one finding that changes how
the central claim should be stated, three claims that were mislabelled or not
reproducible, and one completed experiment from round 8 that had never been analysed.
Two new mechanisms follow from the diagnosis and are under test.

---

## 1. The finding: OptiPrime trains on Kim at one tenth weight

**The corpus the authors supplied carries a per-row `weight` column, and OptiPrime
consumes it as a per-row loss weight.** `scripts/pe/1_train.py` passes
`train_ds.df['weight'].values` into `make_loader`, and `reaction/rx_model.py:195` forms
the objective as `(weights * l2_loss).sum() / n_batch`. The same column reaches the
validation loader but is used there only to count non-padding rows
(`rx_model.py:243-253`), so it does not enter the reported validation metric. It is a
training weight and nothing else.

Read from the authors' own 58 CSVs, not our rebuild:

| lab | files | rows | distinct weights | mean |
|---|---:|---:|---:|---:|
| Kim | 36 | 69,635 | **1** | **0.100** |
| Liu | 8 | 74,769 | 713 | 0.965 |
| Schwank | 14 | 174,067 | 2,685 | 0.521 |

Every Kim row carries weight `0.1` **exactly** — a flat per-study factor, not a per-row
precision estimate. Liu and Schwank vary continuously, consistent with a depth-derived
quantity clipped to [0.1, 1].

**Consequence.** Effective training-gradient share (measured on the 297,962 training
rows by `train_pilot.py --row-weights`):

| source | OptiPrime weights | uniform (ours) | share of held-out |
|---|---:|---:|---:|
| Schwank | 73.2% | 58.4% | 0.0% |
| Liu | 24.8% | 22.0% | 44.7% |
| **Kim** | **2.0%** | **19.6%** | **55.3%** |

OptiPrime allocates **2.0% of its training gradient to the study that supplies 55.3% of
the evaluation set.** Any uniformly-weighted model therefore has an advantage on this
benchmark's dominant partition that owes nothing to architecture or objective.

This is the most parsimonious explanation for the +0.0804 Kim margin, and it displaces
the manuscript's previous speculation ("a mechanistic prior tuned to a narrower regime
would generalise least well"), which was never more than consistent with the data.

**What it does not do.** It does not withdraw the comparison. Both models are scored on
identical rows; each was trained under its own recipe; and uniform weighting is both the
default a practitioner would adopt and the choice our own sweeps support — round 6 found
re-weighting toward the evaluated distribution harmful, and training on the evaluated
sources only costs −0.0294. What changes is the *attribution*: the pooled margin is a
genuine like-for-like result, but the partition structure of that margin is
substantially a data-curation artefact and must not be read as evidence about mechanism
versus architecture.

Caveat stated in the paper: we infer the released checkpoints were trained with these
weights, because the released code consumes this column from the files the authors
supplied. We cannot verify it independently.

**Experiment W1** (`r9_opw`, running): PE-RankFormer trained with OptiPrime's own
per-row weights, against the matched control `r9_ctrl`. This is the weighting-matched
comparison; result pending.

---

## 2. Round 8's selective SSM: the result existed and was never analysed

`r8_sel` and `r8_selfrozen` completed on **30 August**, two days *after* the manuscript
was first committed (28 August), and appear nowhere in it. Recovered from
`results/runs/*/run_info.json`:

| run | best val ρ | vs S4D ctrl | wall clock |
|---|---:|---:|---:|
| `r8_ctrl` (S4D baseline) | 0.8982 | — | 10,583 s |
| `r8_sel` (selective) | 0.8827 | −0.0155 | 121,873 s |
| `r8_selfrozen` (frozen selection) | 0.8796 | −0.0186 | 180,864 s |

**The right comparison is `sel` vs `selfrozen`, not `sel` vs `ctrl`.** The frozen-selection
control has identical parameters and recipe with the selection projections held at
initialisation, so it isolates selectivity from both capacity and recipe. On that
comparison selectivity is worth **+0.0031** — below the ≈0.005 resolution.

Both selective runs sit ~0.016 below the S4D baseline *together*, so that deficit is
attributable to the scan recipe (a selective SSM cannot use the FFT convolution, forcing
batch 256, fp32 and ≈12× the training time) rather than to selectivity. The honest
conclusion is narrow: a selective mixer did not pay for itself here. Added to the
negative-results table with that reasoning spelled out.

Also recovered: `r8_srchead` 0.8958 vs its tied control `r8_srctied` 0.8976 → **−0.0018**
(matches the value already in the manuscript), and `r8_dapt` 0.8998 vs `r8_dapt_ctrl`
0.8992 → +0.0006, null.

---

## 3. Two leads closed by measurement, not by training

**Unused covariates: closed.** The corpus carries `PEmax`, `epegRNA`, `MLH1dn`, `NRCH`
and `time`, none of which the model receives. All five are **fully determined** by the
seven context fields it already has: across the 43 distinct context cells, not one cell
shows any of them varying (0 rows). There is no missing categorical context. This
strengthens the manuscript's argument rather than weakening it — the labels genuinely do
not encode the biology, and the corpus admits only 43 distinct experimental contexts.

**Linker sequence: closed.** 3,029 distinct values, but 94% are the empty string, the
value varies within a design in 2 of 139,278 designs, and within-context demeaned
efficiency is flat across linker lengths. Collinear with context; no signal.

---

## 4. New mechanism C1: context-conditioned non-diagonal conditioning

Round 7 established by a *falsified prediction* that FiLM is the wrong instrument: it
predicted layerwise FiLM would pull the model's cross-condition rank correlation toward
the true 0.683, and it went the other way (0.828 → 0.875). The explanation given was
that FiLM is a per-channel scale and shift, so extra conditioning capacity is spent on
the cross-condition mean, which explains more variance and is easier to fit.

Round 7 then pivoted to the *objective* (ctx-primary, −0.0235, harmful). **It never built
a better instrument.** C1 does:

    h' = h + U diag(a(c)) V^T h

a rank-r correction whose per-mode gains depend on context while the subspaces are
shared. FiLM conditions through a diagonal map and so can only reweight existing
coordinates; this is non-diagonal, so the readout direction presented to the head can
differ between conditions by more than a per-channel rescale.

Implementation notes:
- `U` is zero-initialised **after** `self.apply(self._init_weights)`, which would
  otherwise overwrite it. The block is therefore an exact no-op at step 0 and the
  architecture is a strict superset of the late-FiLM baseline — no perturbed-start
  confound.
- Control (`--context-lowrank-control`) feeds the gain network zeros, so every parameter
  and FLOP is retained but `a` becomes a learned constant: pure extra capacity, no
  conditioning. Parameter counts verified identical.
- +55k parameters on 24.7M.
- 11 unit tests (`tests/test_context_lowrank.py`).

**Correction to the manuscript's phrasing.** An earlier draft of the argument said FiLM
"cannot reorder". That is too strong and a unit test asserting it failed, correctly:
a score `w^T diag(1+γ(c)) h` reweights the channel-wise differences between two designs
and *can* flip their order. The true structural statement, which the tests now pin, is
that FiLM's conditioning is **diagonal** — it cannot form new directions in
representation space — while C1's is not.

**Falsifiable prediction, recorded before the runs finished:** if the mechanism is right,
C1 should *lower* the model's cross-condition rank correlation toward the empirical
0.683–0.758 while raising Kim. A Kim gain without that movement means the gain came from
somewhere else.

Runs: `r9_clr16` (r=16) vs `r9_clr16c` (capacity control) vs `r9_ctrl`. Pending.

## 5. New mechanism C2: the measured zero is censored

The round-7 ceiling analysis established that **21.4% of rows measured at exactly zero
have a non-zero replicate**, and it used that to correct the ceiling. Nothing then used
it in *training*. The ordinal head still supervises a zero row as if every cumulative
indicator were reliably zero.

Measured from the 1,162 symmetrised exact replicate pairs, the replicate distribution
given an observed zero has quantiles:

| q50 | q75 | q90 | q95 | q97.5 | q99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.00159 | 0.00307 | 0.00852 | 0.01095 | 0.0496 |

C2 drops, from a zero row's loss, the threshold terms below a censoring limit — the
indicators the assay cannot support. Terms are **dropped, not down-weighted**, for the
same reason `simplex_loss` marginalises an unmeasured indel rate instead of imputing it.
This differs from the round-6 hurdle head (−0.0022), which models P(y>0) and therefore
still treats the observed zero as a real zero.

Sweep over the empirical quantiles (p90/p95/p99 → 3/3/4 of 18 thresholds masked), plus a
**shuffle control** that drops the same *number* of terms per zero row at random,
matching gradient sparsity without the censoring structure. 7 unit tests
(`tests/test_censored_ordinal.py`), including one asserting no gradient reaches a masked
term.

Prior: moderate. The gap to ceiling scales with zero-fraction (0.18 at 64% zeros vs 0.09
at 38%), which is where this acts. Against it: removing supervision that distinguishes
zero from tiny-positive could hurt the zero-block discrimination Spearman rewards. Which
is why it has a control.

**Wave 2 did not launch as planned.** All five runs OOM'd on the 11GB cards: bf16 is
emulated on Turing, so activations are held at fp32 and the batch-256 recipe needs
>10.5GB against 6.8GB profiled on the RTX PRO 6000. Rather than introduce a
precision-and-batch confound into a wave measuring a sub-0.01 effect, wave 2 waits for
GPUs 6/7 and runs at the standard batch-512 recipe.

---

## 6. A feature-based baseline, to place both models

The manuscript compared against exactly one external model. Added: gradient-boosted
trees (sklearn `HistGradientBoostingRegressor`, no new dependency) on the 17 engineered
features in `family_c_features.parquet` plus the same seven context fields, under the
identical five-fold protocol.

Given every advantage — four hyperparameter settings × two target parameterisations,
winner selected on *held-out* Spearman, the only model here allowed to tune on the test
set:

| model | all | Liu | Kim | within-condition |
|---|---:|---:|---:|---:|
| GBM on engineered features | 0.7413 | 0.6347 | 0.5745 | 0.5460 |
| OptiPrime | 0.8690 | 0.8365 | 0.7320 | 0.7605 |
| PE-RankFormer | 0.9079 | 0.8585 | 0.8124 | 0.8111 |

engineered features **<** mechanistic reaction model **<** learned sequence
representation. The sequence model earns its parameters (+0.167 pooled, +0.265 within
condition), and OptiPrime's reaction structure is doing real work relative to
hand-engineered summaries (+0.128) — which is the more informative way to read our
margin over it.

---

## 7. Manuscript corrections

| # | Issue | Resolution |
|---|---|---|
| 1 | Table 6 labelled "all Kim rows" but was **development fold 0 alone** (n=19,747) | Recomputed on all three dev folds *and* the held-out set (`noise_ceiling_surfaces.py`). Headroom is +0.1149 (dev, 3-fold mean) and +0.1043 (held-out) |
| 2 | Ceiling section compared a dev-fold model score (0.7869) against Table 1's held-out Kim (0.8124) without saying so | Both surfaces now reported and labelled; abstract's "+0.12" → "+0.10" |
| 3 | "28.4% of rows exactly zero" used to motivate a *training* objective; it is the dev-fold figure | Training 16.0%, held-out 26.8%, dev 28.4% — each now labelled at point of use. Kim zero-mass 49.9% (train) vs 50.7% (dev) reconciled |
| 4 | ordinal+S4D = 0.9082 appears as both a dev and a held-out score | Footnote: coincidence, verified distinct at the 5th decimal on disjoint rows (`round6_ordssm_metric_audit.md`) |
| 5 | Resolution quoted as both ±0.002 and ≈0.005 | Standardised on ≈0.005 with a new §"Resolution of the development evaluation" giving the two calibrations. The interaction term (−0.0043) is demoted to a direction |
| 6 | "Held-out set — evaluated once" contradicted a table showing three generations | Corrected to four accesses (audit log), with the adaptive-selection exposure stated and bounded |
| 7 | Empty `\author`; placeholder `hsu2026` citation | Real citation added (Hsu A, … Liu DR, *Nat Biotechnol* **44**, 2026, doi:10.1038/s41587-026-03261-7). **Author list still a TODO — only the authors can fill it** |
| 8 | "eight-fold MAE reduction" against a meaningless baseline | Replaced with the head-to-head: MAE 0.0478 vs 0.0590, RMSE 0.0912 vs 0.1026, Pearson 0.8637 vs 0.8270 — a second axis of superiority the paper had left unclaimed |
| 9 | Early-stopping survey ("25 runs, median 21, max 28, 36%, cost 0.0012") not reproducible | Replaced with a defined, scripted population: all 64 official-fold runs at full budget → median 24.5, max 29, 50% in the final five, patience-5 costs mean 0.0001 / max 0.0031 |

**Verified as correct and unchanged:** every headline number against
`results/round4/final_bootstrap.json`; Table 1's corpus composition against the parquet
(58.4/22.0/19.6% shares, 10.3/0.8/49.9% zero-mass, 16.0% total); all of Table 5 (five
member scores × three partitions, plus all five drop-one marginals) to four decimals;
the 214-test claim (`pytest --collect-only` → 214, now 232); and the bidirectional S4D
padding argument — `flip → causal conv → flip` yields the exact anti-causal convolution
regardless of where right-padding sits, so the no-leak claim holds.

## 8. New robustness result: the margin survives stratification

The pooled 0.9079 exceeds both partition scores, and the model receives `source_study`
as an input, so pooling inflates the *absolute* level. It does not inflate the *margin*:

| comparison | strata | OptiPrime | PE-RankFormer | Δ |
|---|---:|---:|---:|---:|
| pooled | 1 | 0.8690 | 0.9079 | +0.0389 |
| within source | 2 | 0.7788 | 0.8331 | +0.0543 |
| **within condition** | 14 | 0.7605 | 0.8111 | **+0.0506** |

Within-condition Δ = +0.0506, 95% CI [+0.0451, +0.0564] over 5,000
protospacer-clustered resamples, ahead in 100% of them and in **14/14** conditions.
Pooling makes the reported difference conservative. (`stratified_comparison.py`)
