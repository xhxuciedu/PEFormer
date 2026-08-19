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
