# Round-3 Research Log

Chronological experiment log per `claude_code_round3_pe_rankformer_experiments.md` §32.
Each entry: hypothesis, exact change, matched-dev result, Liu result, Kim result,
whether the hypothesis was supported, decision, next experiment.

---

## 2026-08-18 — Phase 0: inventory, baseline freeze, Liu+Kim-matched dev folds

**Hypothesis**: N/A (setup phase).

**Exact change**:
- `reports/round3_initial_inventory.md`: commits, corpus, architecture, existing
  checkpoints and which have known held-out scores (only round-1 baseline and
  round-2 Family C; Family A and D were Stage-A-only in round 2 and were never
  evaluated on held-out data).
- `configs/round3/baseline_round1.yaml`: frozen, byte-identical to round-1's
  actual winning config.
- `scripts/data/build_round3_dev_folds.py`: 3 Liu+Kim-matched, protospacer-
  disjoint development folds from the 297,962 training rows. Design: repeated
  random subsampling (not an exhaustive partition -- the 3 folds' validation
  sets may overlap each other, standard for repeated holdout), row-count-
  weighted (not protospacer-count-weighted -- Liu averages 65.8 rows/protospacer
  vs. Kim's 38.7, so a naive protospacer-count split skewed ~57% Liu instead of
  the target 44.7%; fixed by selecting protospacers via cumulative row count).
  Any protospacer also touching Schwank (62 Liu/Schwank + 241 Kim/Schwank
  collisions found) is pinned to always-train, since Schwank is otherwise
  unconditionally in every fold's training set and a shared protospacer sitting
  in both train (via its Schwank copy) and val (via its Liu/Kim copy) would be
  a real leak. Result: all 3 folds hit 44.6-44.7% Liu / 55.3-55.4% Kim (target:
  44.74%/55.26% exactly), verified protospacer-disjoint, zero Schwank in any
  validation set. 14 regression tests added (`tests/test_round3_dev_folds.py`).
- `scripts/train/train_pilot.py`: added `--dev-folds-file`/`--dev-fold-col` so
  training/evaluation can use the new dev folds instead of val_fold/train_folds.
  Smoke-tested (1 epoch, clean run).

**Development result**: N/A (infrastructure). 126/126 tests passing.

**Hypothesis supported**: N/A.

**Decision**: proceed to Stage 0's re-scoring sanity check (§6) -- evaluate the
4 existing checkpoints (round-1 baseline, round-2 Family A/C/D) on the new dev
folds and check whether the new ranking better predicts the 2 known held-out
outcomes (round-1 > Family C on held-out, despite Family C > round-1 on the
old Schwank-heavy validation).

**Next experiment**: Stage 0 sanity check (§6), then Phase 1 domain adaptation
(§8) if the sanity check passes.

---

## 2026-08-18 — Stage 0 sanity check (§6): INCONCLUSIVE, and it reframes the round

**Hypothesis** (§6): Liu+Kim-matched dev folds should rank round-1 above round-2
Family C, matching the known held-out ordering (0.8865 vs 0.8831), where the old
Schwank-heavy CV ranked them backwards (0.9192 vs 0.9202).

**Exact change**: `scripts/evaluate/evaluate_on_devfolds.py`, evaluating existing
checkpoints on the 3 matched dev folds.

**A methodological bug caught first**: evaluating the round-1 5-model ensemble on
the dev folds gave 0.9431 -- wildly above its known held-out 0.8865. Cause: each
ensemble member trains on 4 of the official folds 1-5, and the dev folds are drawn
from those same folds, so *every* dev row is ~80% in-sample. Added `--oof` mode
(score each row only with the checkpoint that held that row's official fold out).
OOF round-1 dev = 0.8798, now consistent with its 0.8865 held-out. Any dev-fold
number for an official-fold-trained checkpoint that isn't OOF is meaningless.

**Matched-dev result (OOF)**:

| Model | dev combined | dev Liu | dev Kim | held-out combined |
|---|---:|---:|---:|---:|
| round-1 baseline | 0.8798 | 0.8000 | 0.7517 | 0.8865 |
| round-2 Family C | 0.8816 | 0.8130 | 0.7534 | 0.8831 |

Matched dev says Family C is **+0.0018 better**; held-out says it is **-0.0034
worse**. The new validation reproduces the *same wrong ordering* as the old one.

**Diagnosis (paired protospacer-clustered bootstrap, per dev fold):**

| Dev fold | observed Δ (FamilyC − round1) | 95% CI | p |
|---|---:|---|---:|
| 0 | +0.0072 | [+0.0007, +0.0142] | **0.029** |
| 1 | +0.0014 | [−0.0044, +0.0073] | 0.664 |
| 2 | −0.0031 | [−0.0092, +0.0034] | 0.346 |
| *held-out (round 2)* | *−0.0034* | *[−0.0084, +0.0014]* | *0.18* |

The three folds **disagree with each other in sign**, and fold 0 produces a
nominally significant result in the direction opposite to held-out. This is not
a validation-design failure -- it is a **power** failure. The true round-1 vs
Family C difference is ~0.003 or smaller, and *no* evaluation set of this size
(dev or held-out, ~20-36k rows, ~750-800 protospacer clusters) can resolve it.
Note the held-out comparison was itself non-significant (p=0.18); round 2's
"regression" and this round's "improvement" are the same null result read twice.

**Hypothesis supported**: **No, and the check cannot be made to pass with this
model pair.** §6 says "if it does not, diagnose before proceeding" -- diagnosis
is that the check is underpowered by construction, not that the dev folds are
broken. The structural argument for matched dev folds (58% Schwank in training
vs 0% in test) is unchanged and still sound; they simply cannot be *validated*
against a model pair whose true difference is indistinguishable from zero.

**Decision -- this changes round-3 strategy.** The operative lesson from round 2
is not "the validation distribution was wrong" but "**a ±0.003 effect is not
worth chasing, because nothing in this project can measure it.**" Round 3 should
only pursue changes with expected effect sizes well above the ~0.005-0.01
resolution floor, and should require consistency across all 3 dev folds rather
than a mean. Concretely:
- Drop the low-yield end of the spec's plan (adapter-rank sweeps r=8/16/32,
  β-calibration variants, feature-branch revival) unless something large lands
  first.
- Prioritise the one intervention that plausibly moves 0.01+: **domain
  adaptation**, i.e. exploiting the fact that 58% of training is a source that
  is 0% of the target. That is a distributional intervention, not a
  ±0.003 architectural tweak.

**Next experiment**: Phase 1 domain adaptation (§8), designed so it can be
evaluated cleanly: train a global model *per dev fold* (so the dev fold's
validation rows are never in any training stage), then fine-tune on that fold's
Liu+Kim rows only. Requires new `--init-from` and `--train-sources` support in
the training script.

---

## 2026-08-18 — Ensemble DIVERSITY beats architectural improvement (unplanned, decisive)

**Hypothesis** (not in the round-3 spec's plan; came from the Stage-0 diagnosis):
if round-1 and round-2 Family C are statistically indistinguishable *individually*
(true Δ ~0.003, unresolvable), they may still make **different errors** -- in which
case blending them is worth far more than the difference between them. Round 2
treated them as competitors and picked one; that may have been the wrong frame.

**Exact change**: none to any model. Blended the already-computed OOF dev
predictions of the round-1 5-model ensemble and the Family C 5-model ensemble,
via simple mean and via rank-average.

**Matched-dev result (OOF, all 3 folds):**

| Dev fold | round-1 | Family C | mean blend | **rank-avg blend** |
|---|---:|---:|---:|---:|
| 0 | 0.8759 | 0.8831 | 0.8885 | **0.8899** |
| 1 | 0.8820 | 0.8834 | 0.8911 | **0.8926** |
| 2 | 0.8814 | 0.8783 | 0.8893 | **0.8903** |
| **mean** | 0.8798 | 0.8816 | 0.8896 | **0.8909** |

Rank-average blend gains **+0.0093 over the best single model**, and beats mean
blending on every fold. Paired protospacer-clustered bootstrap vs. the better
single model, per fold:

| Dev fold | observed Δ | 95% CI | bootstrap wins | p |
|---|---:|---|---:|---:|
| 0 | +0.0068 | [+0.0033, +0.0103] | 100% | <0.0001 |
| 1 | +0.0092 | [+0.0061, +0.0124] | 100% | <0.0001 |
| 2 | +0.0088 | [+0.0057, +0.0123] | 100% | <0.0001 |

**Hypothesis supported: emphatically yes.** Consistent in sign and magnitude
across all 3 folds, CI excludes zero everywhere, 100% bootstrap wins. Compare to
the round-1-vs-FamilyC difference this same machinery could *not* resolve
(p=0.03/0.66/0.35, signs disagreeing): the blend effect is ~3x larger and
qualitatively different in reliability. Inter-model prediction correlation is
0.909 -- correlated enough to be individually similar, decorrelated enough that
averaging cancels error.

**Why this matters more than the planned experiments**: round 2 spent an entire
round trying to find a *better single architecture* and moved the needle by an
unmeasurable ~0.003. Ten minutes of blending two *already-trained* models moved
it by a reliably-measurable ~0.009. The dominant axis of improvement here is
error decorrelation, not model quality.

**Decision -- reprioritise round 3.** Ensemble diversity becomes the primary
strategy (spec §25, promoted from last to first). Concretely:
1. Round-1 + Family C rank-average is already a validated finalist requiring
   zero new training -- both 5-checkpoint sets exist.
2. Add more *architecturally distinct* members trained across all 5 official
   folds, to extend the effect: Family A (layerwise context) currently has only
   its val_fold=1 checkpoint from round-2 Stage A; folds 2-5 are worth training.
3. Domain-adapted models (Phase 1, in flight) are valuable both on their own
   merits *and* as a further-decorrelated ensemble member -- a
   Liu+Kim-specialised model should make systematically different errors from a
   Schwank-heavy-trained one.

Extrapolating (with appropriate caution -- ensemble gains saturate): if a
2-architecture blend gives +0.009 over its best member, a 3-4 architecture blend
plausibly lands round-1's 0.8865 held-out near 0.895-0.90, which would clear the
spec's "Strong" bar vs OptiPrime (Δρ ≥ 0.025 vs the current +0.0175) and
approach "Excellent" (ρ_full ≥ 0.90). Not assumed -- to be measured on dev folds
before any held-out query.

**Next experiment**: Phase 1 domain adaptation (global models finishing now),
then Family A official folds 2-5 to add a third diverse ensemble member.
