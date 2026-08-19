# Round-4 Research Log

Per `claude_code_round4_pe_rankformer_ensemble_search.md` §29. Each entry: hypothesis,
why the model may be complementary, implementation, standalone result, diversity
result, ensemble result, decision, independent observations, next step.

The round-4 search criterion is **incremental ensemble gain S3**, not standalone
Spearman (§5, §32).

---

## 2026-08-19 — Phase 0: three-level evaluation hierarchy + diversity framework

**Hypothesis**: N/A (infrastructure).

**Implementation**:

1. **Round-4 internal lockbox** (`scripts/data/build_round4_lockbox.py` ->
   `data/processed/round4_lockbox.parquet`). 17,975 rows, 367 protospacers,
   44.8% Liu / 55.2% Kim -- matching the official held-out set's 44.74%/55.26%
   by row-count-weighted selection (protospacer-count weighting skews Liu-heavy;
   same trap as round 3's dev folds).

   Eligibility: Liu+Kim protospacers that never appear in **any** round-3
   dev-fold validation set, and that do not also occur in Schwank (which is
   unconditionally training data, so a shared protospacer would straddle train
   and lockbox). 646 protospacers / 37,059 rows were eligible; 367 selected to
   hit the target size and ratio.

   **The one real subtlety, stated plainly**: lockbox rows come from the 297,962
   training rows, so any model trained on the official 5-fold split *has* trained
   on lockbox rows lying in folds other than its own held-out fold. The lockbox
   must therefore be scored **out-of-fold**, exactly like the dev folds. What
   makes it a lockbox is not stronger data isolation but a *usage* guarantee: it
   is disjoint from every dev-fold validation set, so nothing has ever been
   selected, early-stopped, or weighted on these rows. Verified it spans all 5
   official folds so OOF scoring can cover every row. 6 regression tests.

   Resulting hierarchy: dev folds (free use) -> lockbox (once, to screen the
   shortlist) -> official held-out (once, after freeze).

2. **Diversity framework** (`scripts/evaluate/diversity_report.py`). Computes the
   four §5 quantities per candidate: S1 standalone, S2 prediction-rank
   correlation with the current ensemble, S2r residual correlation, and S3
   incremental ensemble gain under the frozen equal-weight rank-average rule.

   Design note: residuals are computed in **rank space**. The ensemble output is
   a rank average and its members are not mutually calibrated, so a
   raw-efficiency residual correlation would largely measure inter-member scale
   mismatch rather than which examples each model actually gets wrong.

**Validation of the framework itself**: ran it on a case with a known answer --
round-1 as a candidate against the frozen round-3 ensemble. Result: S2 = 0.982
(highly redundant), S3 = **-0.0017**, negative on all 3 folds. This independently
reproduces round 3's separately-established finding that adding round-1 back
significantly hurts, which is a reassuring check that S3 is measuring what it
should.

**Decision**: proceed to Phase 1. Priority order chosen on expected
*decorrelation*, not expected standalone quality, per §32:
- **PE-SSM** (§8) first among the training experiments -- a materially different
  inductive bias (state-space/convolutional rather than attention) is the most
  plausible source of genuinely decorrelated errors, and round 3 showed
  decorrelation is worth ~3x more than standalone improvement.
- **R4-Medium-AdaLN** (§7) in parallel -- larger capacity plus adaptive-LayerNorm
  conditioning; a real risk here is that it lands close to Family A (also
  layerwise-conditioned), which S2/S3 will reveal before any 5-fold spend.
- Zero-training ensemble experiments (§9 context-gated, §10 stacking) run
  alongside on CPU, since they need no GPU and use existing OOF predictions.

**Next**: implement PE-SSM and R4-Medium-AdaLN; screen both on one dev fold
before committing 5-fold compute (§19 successive halving).

---

## 2026-08-19 — Context-gated ensembling (§9) and nonlinear stacking (§10): both negative

**Hypothesis** (§9): round 3 showed *global* fitted weights were unstable and never
beat equal weighting. Context-*dependent* weights are a narrower, better-posed
question -- maybe Family C is the right member to upweight on Kim rows and Family A
on Liu rows, even if no single global weighting helps. (§10): a nonlinear stacker
over OOF member predictions plus context might capture interactions that a linear
rank average cannot.

**Why these might be complementary**: neither requires new training -- both reuse the
existing OOF predictions -- so they are the cheapest possible sources of gain.

**Implementation**: `scripts/evaluate/context_gated_ensemble.py`. Gates map one-hot
(source, cell type, PE system) to simplex weights, initialised at exactly equal
weights so any departure has to be earned, and regularised by
lambda * sum_k (w_k - 1/K)^2. Stackers are ridge and histogram-GBM over
[member ranks ++ context dummies]. Evaluated by rotating: fit on two dev folds,
score on the third.

**A leakage bug in my own first run, caught and fixed.** The initial version
reported GBM stacking at **0.9012 (+0.0030)** -- above the 0.90 target and above the
§6 promotion threshold. Before acting on it I checked whether the "nested" split was
really nested. It was not: the three round-3 dev folds are *repeated random
subsamples*, not a partition, and **57-60% of each fold's rows also appear in the
other two**. The stacker was fitting on the majority of its own test rows. Refitting
with the fitting set filtered to exclude every row whose **protospacer** occurs in
the scoring fold (protospacer-level, since rows sharing a protospacer are strongly
correlated) drops the fit set from ~58.5k to ~38.1k rows and gives the honest
numbers below.

**Result (nested, protospacer-disjoint):**

| Method | mean | per-fold | vs equal-rank |
|---|---:|---|---:|
| stack_gbm | 0.8995 | 0.8998 / 0.9012 / 0.8977 | +0.0013 |
| stack_ridge | 0.8990 | 0.8995 / 0.9006 / 0.8971 | +0.0008 |
| **equal_rank (frozen rule)** | **0.8982** | 0.8979 / 0.8997 / 0.8969 | — |
| gate_mlp | 0.8982 | 0.8980 / 0.8998 / 0.8970 | +0.0000 |
| gate_linear | 0.8982 | 0.8980 / 0.8998 / 0.8969 | +0.0000 |
| source_weights | 0.8980 | 0.8977 / 0.9000 / 0.8963 | −0.0002 |
| equal_score | 0.8960 | 0.8956 / 0.8974 / 0.8949 | −0.0022 |

**Hypothesis supported: no, for both.**

- **Context gating is a clean null** (+0.0000). Both the linear and MLP gates,
  initialised at equal weights and free to move, essentially stay there. Source-
  specific fixed weights are also null (−0.0002). This now extends round 3's
  finding: it is not just that *global* weight fitting is unstable -- there is no
  recoverable signal saying "member k is better in context c" at all. The members
  differ in *which rows* they get wrong, not in *which contexts* they are good at.
  That is a genuinely useful negative: it means diversity here is row-level, and
  ensembling can only exploit it by averaging, not by routing.
- **Stacking is positive but below threshold** (+0.0013 GBM, +0.0008 ridge; below
  §6's +0.003 bar and below the ~0.005 resolution floor established in round 3),
  though positive on all 3 folds. Not promoted now; worth one re-test if new
  members materially change the ensemble, since a stacker's value should grow with
  member count.
- **equal_score is −0.0022 vs equal_rank**, independently reconfirming the frozen
  rank-average rule.

**Decision**: keep equal-weight rank averaging. Do not pursue gating further.
Re-test GBM stacking once (and only if) PE-SSM or the medium model joins the
ensemble.

**Independent observation worth recording**: the 57-60% dev-fold overlap is a
latent hazard for *any* round-4 experiment that fits something on dev-fold
predictions (stacking, gating, calibration, residual learning). The dev folds were
built in round 3 as repeated holdouts for *evaluating* models, where overlap is
harmless; they are not a partition and must not be used as one. Any future fitting
on these folds must apply the same protospacer-disjoint filter.
