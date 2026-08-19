# Round 3 Results: Ensemble Diversity Beats Architecture Search

Per `claude_code_round3_pe_rankformer_experiments.md`. Full experiment log:
`reports/round3_research_log.md`. Frozen model spec:
`reports/round3_final_model_spec.md`.

## 1. Objective

Correct round 2's validation-composition mismatch (58% Schwank in every CV fold,
0% in the held-out set) and use it to push predictive performance clearly beyond
round 1 and OptiPrime on the official 20,509-row Liu+Kim held-out benchmark.

## 2. Why round-2 validation failed

Every round-1/round-2 CV fold was drawn uniformly from all 297,962 training rows,
58.4% of which are Schwank. The held-out set contains **zero** Schwank rows (Liu
44.7% + Kim 55.3%, exactly). Round 2's confirmed, 5-fold-stable validation gain
(+0.0028, std 0.0011) inverted sign on held-out (-0.0034) -- the validation target
and the test target were measuring different things.

## 3. New Liu+Kim-matched validation design

`scripts/data/build_round3_dev_folds.py`: 3 protospacer-disjoint development
folds via repeated random subsampling from the 297,962 training rows, row-count-
weighted (not protospacer-count-weighted -- Liu averages 65.8 rows/protospacer
vs. Kim's 38.7, so naive protospacer-count splitting skewed 57% Liu instead of
the 44.7% target). Any protospacer also appearing in Schwank (303 collisions) is
pinned to always-train. All 3 folds hit 44.6-44.7% Liu / 55.3-55.4% Kim (target
44.74%/55.26% exactly), verified protospacer-disjoint, zero Schwank in validation.

## 4. Re-evaluation of round-1/round-2 models -- the check that reframed the round

Evaluating existing checkpoints on the new dev folds first required an **out-of-
fold (OOF)** mode: naively ensembling the 5 official-fold checkpoints on dev-fold
rows gave 0.9431 (vs. round-1's known 0.8865 held-out), because every ensemble
member trains on 4 of the 5 official folds the dev folds are drawn from --
~80% in-sample. Fixed by scoring each row only with the checkpoint that held that
row's official fold out; OOF round-1 dev = 0.8798, consistent with held-out.

**The corrected validation still ranked round-2 Family C above round-1**
(0.8816 vs. 0.8798) -- the same wrong ordering as the old Schwank-heavy CV, vs.
the known held-out ordering (round-1 0.8865 > Family C 0.8831). Diagnosed via
paired protospacer-clustered bootstrap per fold rather than accepted at face
value: the three dev folds disagreed in *sign* (+0.0072 p=0.029, +0.0014 p=0.66,
-0.0031 p=0.35), and the held-out comparison was itself non-significant
(p=0.18). **Conclusion: the true round-1-vs-Family-C difference is ~0.003 and
unresolvable by any evaluation set available in this project** (~750-800
protospacer clusters). This was not a validation-design failure -- the
structural argument for matched folds is untouched -- but it reframed the
round's strategy: stop chasing sub-0.005 single-model effects.

## 5. Domain-adaptive fine-tuning

Fine-tuned the round-1 official-fold checkpoints on Liu+Kim only (`--init-from`,
lr 3e-5, 10 epochs, `--val-sources` restricting checkpoint selection to Liu+Kim).
**+0.0015 mean, positive on all 5 folds** (+0.0006 to +0.0027) vs. each
checkpoint's own pre-fine-tune control. Real but small.

## 6. Schwank replay

Implemented (`--schwank-replay-frac`) and smoke-tested but not swept -- deferred
once the ensemble-diversity finding (§9) redirected priority. Open for a future
round.

## 7. Source-balanced training

Not attempted. Deferred for the same reason as §6.

## 8. Layerwise context conditioning, revisited

Round 2 found this helped its own (Schwank-heavy) validation but never reached a
held-out evaluation. Retrained on official folds 2-5 with Liu+Kim-only
validation (fold 1 from round-2 Stage A was discarded and retrained, since it
had been selected on Schwank-inclusive validation and was not comparable).
**+0.0041 mean, positive on 3 of 4 folds** (+0.0048, +0.0037, +0.0083, -0.0004)
-- the largest single-model effect measured across rounds 2-3, nearly 3x domain
adaptation's effect.

## 9. Source-specific adapters

Not attempted. The ensemble-diversity finding (below) was discovered before
reaching this planned phase and was large enough to redirect all remaining
effort toward it.

## 10. Liu/Kim specialized models

Not attempted, superseded by the ensemble finding.

## 11. Calibration

Not needed: the frozen ensemble combines members by rank-average (fractional
rank in [0,1]), which is invariant to any per-member calibration/scale
difference by construction.

## 12. Model scaling

Not attempted this round -- gated behind domain adaptation "working
well" per the spec's own ordering, and superseded once ensembling proved to be
the dominant lever.

## 13. Self-supervised pretraining

Not attempted this round.

## 14. Finalists -- the central finding of round 3

**Ensemble diversity, not any single architectural change, is the dominant
lever.** Blending the OOF dev predictions of already-trained models:

| Configuration | Mean dev Spearman | Δ vs. round-1 |
|---|---:|---:|
| round-1 (single) | 0.8798 | -- |
| Family C (single) | 0.8816 | +0.0018 |
| DAPT (single) | 0.8820 | +0.0022 |
| **Family A (single)** | **0.8845** | **+0.0047** |
| Family C + DAPT (2-way) | 0.8919 | +0.0121 |
| **Family C + DAPT + Family A (3-way)** | **0.8982** | **+0.0184** |

The mechanism: rank-prediction correlation between round-1 and DAPT is **0.997**
(fine-tuning barely moves the function -- DAPT supersedes round-1 rather than
adding to it), while Family A correlates only **0.945-0.947** with every other
member (the most architecturally distinct model, explaining why it is both the
best standalone model and the best ensemble addition). Verified with paired
bootstrap that the 3-way blend significantly beats the best 2-way (p<0.0001 on
all 3 folds) and that **adding round-1 back significantly hurts** (p<0.0001,
p<0.0001, p=0.034) -- not an artifact of searching many subsets.

Fitted (non-equal) ensemble weights were tested and rejected: never beat equal
weighting, and the fitted optimum was unstable fold-to-fold.

**Finalist**: Family C + DAPT + Family A, rank-average, equal weights, round-1
excluded.

## 15. Full 5-fold CV

Every member is already a 5-checkpoint official-fold ensemble (Family C reused
from round 2; DAPT and Family A trained fresh this round on all 5 official
folds). 15 checkpoints total, all verified present before freezing.

## 16. Ensemble construction

Combination rule selected on matched dev folds: rank-average beat mean-average
on every fold tested; fitted weights never beat equal weights. Final rule:
convert each member's held-out predictions to fractional rank, average the 3
members' ranks equally.

## 17. Frozen final model

Commit `692274b`. Full spec: `reports/round3_final_model_spec.md`.

## 18. Official held-out evaluation

Full 20,509-row held-out set, Liu (9,175) and Kim (11,334) partitions.

| Scope | OptiPrime | Round-1 | Round-2 | **Round-3 (frozen)** |
|---|---:|---:|---:|---:|
| Full | 0.8690 | 0.8865 | 0.8831 | **0.8933** |
| Liu | 0.8365 | 0.8349 | 0.8298 | **0.8462** |
| Kim | 0.7320 | 0.7751 | 0.7727 | **0.7836** |

Round 3 improves on **every** scope over **every** prior round -- the first time
across all three rounds that a change has not traded off Liu against Kim or
validation against held-out.

Per-condition: round-3 beats OptiPrime on **10/10 Kim conditions** (n>=50 each),
by margins of +0.023 to +0.099 Spearman.

**Note on Pearson/MAE/RMSE**: the frozen ensemble's output is a rank-averaged
score in [0,1], not a calibrated efficiency estimate (see §16), so Pearson r,
MAE, and RMSE are not meaningful for the blend and are not reported at the
ensemble level. They remain meaningful, and are reported, per individual member
(`results/round3/heldout/metrics_r3_final_ensemble.json`, `member_only_global`).
Spearman -- the primary metric throughout this project -- is rank-invariant and
unaffected by this.

## 19. Comparison with OptiPrime

Δρ = **+0.0243** on the full set, up from round-1's +0.0175 and round-2's
+0.0141. On Kim, Δρ = +0.0515 (round-1: +0.0431). On Liu, round-3 is nominally
*ahead* of OptiPrime for the first time across any round (+0.0098), though not
significantly (§20) -- previously round-1 and round-2 were tied-or-behind.

## 20. Statistical significance

Paired, protospacer-clustered bootstrap, 5000 resamples.

**Round-3 vs. round-1:**
- Full (750 clusters): obs +0.0068, 95% CI **[+0.0042, +0.0094]**, 100% wins,
  **p<0.0001**.
- Liu (150 clusters): obs +0.0113, 95% CI [+0.0026, +0.0199], 99.5% wins,
  **p=0.01**.
- Kim (601 clusters): obs +0.0085, 95% CI [+0.0049, +0.0123], 100% wins,
  **p<0.0001**.

**Round-3 vs. OptiPrime:**
- Full: obs **+0.0243**, 95% CI **[+0.0142, +0.0345]**, 100% wins, **p<0.0001**.
- Liu: obs +0.0098, 95% CI [-0.0108, +0.0303], 83.7% wins, p=0.33 (not
  significant -- statistically tied, as in prior rounds, but no longer trending
  negative as round-2 did).
- Kim: obs **+0.0515**, 95% CI **[+0.0348, +0.0681]**, 100% wins, **p<0.0001**.

Round 3 is a statistically decisive improvement over round-1 on the full set
*and* on both partitions separately -- something neither prior round achieved.

## 21. Negative results

- Fitted (non-equal) ensemble weights never beat equal weighting, on any member
  subset tested; the fitted optimum was unstable across dev folds. (§14)
- Domain adaptation, standalone: real but small (+0.0015 mean), and its
  0.997 correlation with its parent means it cannot usefully co-occur with
  round-1 in an ensemble -- confirmed by direct bootstrap, not just the
  correlation heuristic. (§5, §14)
- Round-1 baseline, included in the final ensemble: **significantly hurts**
  once DAPT and Family A are present (p<0.0001/p<0.0001/p=0.034 across the 3
  dev folds). (§14)
- The Stage-0 sanity check specified by the round-3 plan (§6) could not be made
  to pass with the round-1-vs-Family-C model pair, because that pair's true
  difference is smaller than what any available evaluation set can resolve.
  This is reported as a finding about measurement power, not a failure of the
  matched-dev-fold design.

## 22. Compute cost

Stage 0: 3 dev-fold builds (cheap, CPU) + re-scoring 2 existing 5-checkpoint
ensembles (OOF, ~10 GPU-min each). Phase 1: 5-fold domain adaptation (~2 GPU-h
total, 10-epoch fine-tunes) + one abandoned lr=1e-5 arm (killed after 3/5 folds
once redundancy with lr=3e-5 was established from correlation, not re-run).
Family A: 5 official-fold trainings from scratch (~3 GPU-h, including one
discarded/retrained fold-1 run for validation-scheme consistency). Final
held-out evaluation: 15 checkpoints x ~4s/checkpoint. Total: substantially less
compute than round 2's architecture search, for a larger and statistically
cleaner result.

## 23. Recommendation for Round 4

1. **The dominant lever is ensemble diversity, and it is not exhausted.** Every
   architecturally-distinct model added has helped; every near-duplicate
   (fine-tuned variant) has not. The highest-value next step is training 1-2
   more genuinely different architectures (e.g. moderate scaling, §12/§15 of
   the round-3 spec, or a pretraining-initialized model, §13/§16) specifically
   to test whether they decorrelate like Family A (~0.945) or like DAPT
   (~0.997) before investing in a full 5-fold training run.
2. **Do not chase single-model effects below ~0.005 dev Spearman.** Stage 0
   demonstrated this project's evaluation sets (dev or held-out, ~750-800
   protospacer clusters) cannot resolve effects at that scale, in either
   direction. Any future candidate should be screened for a *plausible*
   effect size before a full run, not just a directionally-favorable one.
3. **Dev-fold scores rank candidates but do not predict held-out magnitude.**
   The dev-to-held-out gap was not stable across architectures in this round
   (+0.0067 for round-1, +0.0015 for Family C) -- useful for ordering
   candidates, not for forecasting the final number.
4. Round 3's untried spec items (source-balanced training, adapters, Liu/Kim
   specialization, scaling, pretraining) remain open and were set aside for
   compute-efficiency reasons once the ensemble finding dominated, not because
   they were tested and failed.
5. The frozen model still does not clear ρ_full >= 0.90 (0.8933, short by
   0.0067) or the letter of the spec's "Strong" Δρ_OptiPrime >= 0.025 bar
   (achieved 0.0243). Both are close enough that 1-2 more decorrelated
   ensemble members could plausibly close the gap.
