# Round-5 initial inventory (spec §3)

Git commit at inventory time: `cd00264`. All numbers are round-4 artifacts, recomputed
from stored predictions rather than quoted from prose.

## Baseline model: Ordinal-SSM

| Property | Value |
|---|---|
| Config | `configs/round5/baseline_ordinal_ssm.yaml` (frozen) |
| Architecture | dual-encoder, bidirectional cross-attention, **bidirectional diagonal S4D** sequence mixer |
| Objective | CORAL-style **ordinal** head, K=20 requested → 17–18 distinct thresholds |
| Context | late FiLM (layerwise not implemented for SSM; guarded) |
| Parameters | 24.7 M |
| Thresholds | quantiles of the **training** targets only, stored in checkpoint |
| Checkpoints | `checkpoints/r4p2_ordSSM_cv{1..5}/best.pt` |
| Training folds | official 5-fold; each checkpoint holds out one fold |

### Verified performance

| Surface | ρ (full) | Liu | Kim |
|---|---:|---:|---:|
| OOF, mean of 3 matched dev folds | **0.9082** | 0.8444 | 0.7912 |
| OOF, round-4 lockbox | 0.9040 | — | — |
| *(reference)* round-3 3-member ensemble, dev | 0.8982 | — | — |
| *(reference)* round-4 5-member ensemble, held-out | 0.9079 | 0.8585 | 0.8124 |

A single Ordinal-SSM out-scores the entire round-3 ensemble on dev.

### Correlation to the round-4 ensemble

Ordinal-SSM is the round-4 ensemble's strongest member; the other four are
`r4p2_ssm`, `r4p2_ordC`, `r4p2_ordA`, `r3_familyA`. Pairwise structure below.

## Objective × architecture factorial (spec §11) — computed from existing artifacts

All four cells already existed as round-4 OOF predictions, so this cost no compute.

| | simplex | ordinal |
|---|---:|---:|
| **Transformer** | 0.8798 | 0.8911 |
| **SSM** | 0.9011 | **0.9082** |

| Effect | Value |
|---|---:|
| Objective (ordinal − simplex), averaged over architecture | **+0.0092** |
| Architecture (SSM − Transformer), averaged over objective | **+0.0192** |
| Interaction | −0.0043 |

### This corrects a claim I made in round 4

The round-4 report described the ordinal head as "the breakthrough… a change of
objective, not architecture." **The factorial does not support that.** The
architecture effect (+0.0192) is roughly **twice** the objective effect (+0.0092).
The ordinal head was the more *novel* contribution and the one that produced
decorrelated errors, but the SSM mixer contributed more raw accuracy. The round-4
write-up over-credited the objective, and the round-5 report will say so.

The interaction is **sub-additive** (−0.0043): ordinal supervision gains +0.0114 on a
Transformer but only +0.0071 on an SSM. The two changes partly capture the same
improvement rather than stacking cleanly — which sets a realistic ceiling on further
"add another mechanism" gains.

Comparability caveat: these are observational cells, not a purpose-built factorial.
Three are round-4 Phase-2 runs; `Transformer/simplex` is the frozen round-1 baseline
trained at an earlier commit. They share corpus, splits, OOF scoring, capacity and
optimiser recipe, so the main effects are trustworthy to roughly the ±0.002 dev
resolution, but the interaction term is the quantity most exposed to that mismatch.

## Diversity structure (fold-averaged, dev)

| Pair | prediction corr | residual corr |
|---|---:|---:|
| Transformer/simplex ↔ **SSM/ordinal** | 0.9334 | **0.6453** |
| Transformer/ordinal ↔ SSM/simplex | 0.9467 | 0.7017 |
| Transformer/simplex ↔ Transformer/ordinal | 0.9436 | 0.7072 |
| Transformer/ordinal ↔ SSM/ordinal | 0.9492 | 0.7170 |
| Transformer/simplex ↔ SSM/simplex | 0.9475 | 0.7352 |
| **SSM/simplex ↔ SSM/ordinal** | **0.9639** | **0.7641** |

Two actionable readings:

1. **The diagonal is the most complementary pair.** Changing *both* axes
   (Transformer/simplex vs SSM/ordinal) gives the lowest residual correlation
   (0.6453) of any pair.
2. **The round-4 ensemble contains a redundancy.** Its two strongest members,
   `ordSSM` and `ssm`, are the *most correlated* pair in the table (0.9639 /
   0.7641) because they share an architecture. That suggests a round-5 ensemble
   lead: swap one for a Transformer-side member of comparable strength. Worth
   testing, but only against the promotion rule — round 4 showed decorrelation
   alone does not earn a slot.

## Evaluation surfaces available to round 5

| Surface | Size | Status |
|---|---|---|
| Matched dev folds (×3) | ~35.7k rows each | freely usable, but worn by rounds 3–4 |
| Round-4 lockbox | 17,975 rows / 367 protospacers, 44.8% Liu | used **once**; composition-matched |
| **Round-5 lockbox** | 19,084 rows / 279 protospacers, **98.5% Liu** | fresh, but **Liu-only** |
| Official held-out | 20,509 rows | LOCKED until final freeze |

### A composition-matched round-5 lockbox is not available

Of 2,501 Liu+Kim training protospacers: 1,553 sit in dev validation sets, 367 were
consumed by the round-4 lockbox, and 302 collide with Schwank. The 279 that remain
are 98.5% Liu.

**Consequence, stated up front: Kim has no fresh gate left.** The round-5 lockbox can
detect Liu-side overfitting and nothing else. Kim-side claims — which is where round 5
most wants to improve, Kim being 0.8124 against Liu's 0.8585 — rest on OOF dev folds
plus the once-used round-4 lockbox, and are the less-protected half of any result.
That is a real limitation of the benchmark, not something to design around.

## Round-5 plan implied by this inventory

1. **Supervision geometry** (running): dual-head A, context-ordinal C, multi-res B.
2. **Architecture** promoted in priority. The factorial says architecture is the
   larger lever and cross-architecture pairs are the most complementary, so the
   hybrid mixer (§12) is elevated above its advisory position.
3. **Ensemble composition** revisited given the `ordSSM`/`ssm` redundancy.
4. Calibration (§15) as a deliverable — the current rank-average has no absolute scale.
