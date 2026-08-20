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

---

## 2026-08-19 — Residual learning (§11): negative, with an informative upper bound

**Hypothesis** (§11): train a model on what the ensemble misses,
r = rank(y) - yhat_E, using the 16 engineered Family-C features plus context --
a genuinely different information source from the members, which see only sequence
and categorical context. Predict yhat = yhat_E + eta * rhat.

**Why this might be complementary** (and why it is not just §10 again): the stacker
reweights *member predictions*, so it can only recombine what the members already
encode. A residual learner sees engineered features directly and could in principle
supply information no member has.

**Implementation**: `scripts/evaluate/residual_learner.py`, HistGradientBoosting on
[16 engineered features ++ context dummies], nested and protospacer-disjoint (same
57-60% dev-fold overlap hazard as §10).

**Result: -0.0040 mean, negative on all 3 folds.** eta was selected as 1.0 on every
fitting fold -- the residual model fit its own training residuals well, and that fit
did not transfer at all.

**But the raw number understates the case, so I bounded it properly.** A better eta
tuner would simply drive eta toward 0 and recover ~0.0000, so "-0.0040" measures my
eta-selection rather than the method. To get the real answer I computed an **oracle
bound**: choose eta *on the scoring fold itself* (an illegitimate procedure, used
here only as a diagnostic ceiling):

| Dev fold | ensemble | best achievable | max gain |
|---|---:|---:|---:|
| 0 | 0.8979 | 0.8984 | +0.0005 |
| 1 | 0.8997 | 0.9007 | +0.0010 |
| 2 | 0.8969 | 0.8973 | +0.0004 |

**Even with oracle eta selection the ceiling is +0.0006 mean** -- an order of
magnitude below the §6 promotion threshold. So this is not a tuning failure: the
ensemble's residual is essentially **unpredictable** from engineered features and
experimental context.

**Hypothesis supported: no, and definitively.**

**Decision**: close §11. Do not attempt the other residual-model variants (small
Transformer head, MLP on frozen embeddings, small SSM) -- they draw on the same
information the GBM already had, and the ceiling applies to the information, not to
the model class.

**Independent observation -- the emerging shape of round 4.** Three post-hoc
approaches have now been tested, all on existing predictions at near-zero cost, and
all fail:

| Approach | § | Result |
|---|---|---|
| Context-gated weights | 9 | +0.0000 (clean null) |
| Nonlinear stacking | 10 | +0.0013 (below threshold) |
| Residual learning | 11 | +0.0006 oracle ceiling |

The consistent message is that **nothing recoverable remains in the current
members' outputs, nor in engineered features or context**. The ensemble's remaining
error is either irreducible measurement noise or requires information no current
model represents. That is a strong argument that the only productive direction left
is genuinely new *models* with different inductive biases -- exactly what PE-SSM
and the medium AdaLN model are testing -- and it justifies not spending further
compute on post-hoc combination machinery this round.

---

## 2026-08-19 — Phase 1: 11-candidate screen. The ordinal head works.

Trained 11 candidate members on dev fold 0 in parallel, all 30 epochs, no failures.
Diversity metrics against the frozen round-3 ensemble (`round2_familyC_oof` +
`r3_dapt_lr3e5_oof` + `r3_familyA_oof`, 0.8979 on this fold):

| candidate | S1 solo | S2 corr | S2r resid | **S3 gain** | verdict |
|---|---:|---:|---:|---:|---|
| **ordinal + SSM** | **0.8972** | **0.9407** | **0.6863** | **+0.0092** | promote |
| PE-SSM | 0.8879 | 0.9499 | 0.7564 | +0.0060 | promote |
| ordinal + features | 0.8856 | 0.9462 | 0.7260 | +0.0054 | promote |
| ordinal + layerwise | 0.8763 | 0.9379 | 0.6907 | +0.0042 | promote |
| ordinal (base) | 0.8803 | 0.9443 | 0.7075 | +0.0040 | promote |
| bagged + ordinal | 0.8696 | 0.9356 | 0.6856 | +0.0029 | borderline, hold |
| medium AdaLN | 0.8724 | 0.9493 | 0.7649 | +0.0022 | reject |
| Family A seed 2 | 0.8711 | 0.9479 | 0.7628 | +0.0022 | reject |
| bagged x3 | 0.854-0.862 | ~0.947 | ~0.77 | +0.0002 / -0.0008 / -0.0022 | reject |

**The ordinal-head hypothesis is confirmed, and confirmed by the right evidence.**
The four lowest residual correlations in the table (0.6856-0.7075) all belong to
ordinal-head models; every simplex-head candidate sits at 0.76-0.78. The prediction
was specifically that a different *loss geometry* would produce different *errors*,
and S2r is the direct measurement of exactly that. It is not a post-hoc story fitted
to a good number.

`ordinal + SSM` is the strongest single result of the project so far: it stacks the
two independent axes (objective and sequence mixing) and lands at both the highest
standalone score and the lowest correlation with the ensemble -- normally a trade-off.
Its +0.0092 alone exceeds the +0.0067 needed to reach 0.90.

**Bagging failed, and the reason is informative.** All four bagged members scored
worst on S1 *and* failed to compensate on S3 (+0.0002 to -0.0022). Dropping 30% of
protospacers cost more accuracy than the induced diversity was worth. With ~38k
training protospacers the models are not in a variance-limited regime where bagging
pays; they are limited by what the architecture and objective can express. This is
consistent with everything else round 4 has found.

**Sub-architecture scale changes are dead ends.** Medium AdaLN (+0.0022) and a
reseeded Family A (+0.0022) both land below the bar -- more capacity and more seeds
do not produce new errors. Only new *mechanisms* did.

### The caveat that matters, stated before the numbers get quoted

**These S1/S3 figures are optimistically biased and must not be compared against
the incumbents' numbers.** Each candidate trained on dev fold 0's training split and
had its best epoch chosen on that same fold's validation rows -- the rows it is then
scored on. The incumbent members are out-of-fold on the official split and carry no
such bias. Best-of-30-epochs selection on 35,649 rows is worth roughly +0.001-0.003,
which is the same order as the effects being measured.

So Phase 1 is a *screen*: it ranks candidates that all share the identical bias, and
that ranking is trustworthy. It does not establish what any member is worth. The
"positive on all folds" clause of the promotion rule is also vacuous here, since only
one fold was run.

For the same reason the round-4 lockbox **cannot** be used to check these
checkpoints: its rows were excluded from every dev *validation* set, which means they
sat in the dev *training* splits, so these models trained on them. The lockbox
remains valid only for models trained on the official split.

**Phase 2 (running): all five promoted members retrained on the official 5-fold
split**, 25 runs, which puts them on exactly the same OOF footing as the incumbents
and makes the Phase-3 comparison honest.

### Process note

The first Phase-2 scheduler assumed ~15GB per run and packed fixed slots; the SSM
runs actually need ~22GB and 9 of 25 jobs OOM'd on launch. Replaced the guess with
`queue_runs.sh`, which reads each GPU's real free memory and dispatches only when a
card can hold the job. It skips runs whose checkpoint already exists, so the 17
missing jobs were requeued without disturbing the 8 still training.

### Wave 2 (pre-registered before results): pushing the objective axis further

Phase 2 only occupies the three large cards, so the five 11GB cards were idle.
Queued four more dev-fold-0 screens there, all on the axis that actually paid --
the training objective:

| run | change | question |
|---|---|---|
| `r4w2_ordrank` | ordinal + `lambda_rank=0.25` | do a cumulative-threshold loss and an explicit pairwise ranking loss supply *different* order information? |
| `r4w2_rank` | simplex + `lambda_rank=0.25` | isolates the ranking loss on its own, so any ordrank gain can be attributed |
| `r4w2_ordK8` | ordinal, 7 thresholds | coarse quantile supervision |
| `r4w2_ordK50` | ordinal, 43 thresholds | fine quantile supervision |

The pairwise ranking loss is the one objective already in the codebase that round 1
tested and dropped (`lambda_rank` 0.25 -> 0.0). It was dropped on *standalone*
grounds, which round 3 showed is the wrong criterion for an ensemble member -- so it
is worth re-testing on S3, where the question is different.

K=8 vs 20 vs 50 is a genuine hypothesis rather than a sweep: the threshold count
controls how much of the target distribution's shape the loss sees. Too coarse and
it degenerates toward binary classification; too fine and each indicator gets few
effective positives. If all three land at the same S3 the head is insensitive to K,
which is itself worth knowing.

**Recipe caveat**: these use `configs/round4/bagged.yaml` (batch 256, lr 2e-4) since
that is the config that fits an 11GB card -- despite the name, `--bag-frac` defaults
to 1.0 so no bagging is applied. Their S1 values are therefore **not** comparable to
the batch-512 Phase-1 screens; only comparisons *within* wave 2, and S3 gains, are
meaningful. Any wave-2 member that gets promoted must keep this recipe across all
five of its folds.
