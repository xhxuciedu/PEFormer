# Round-4 final model — FROZEN

**Frozen 2026-08-20, before any round-4 held-out evaluation.** Selected by the
protocol pre-registered in `round4_plan_v2.md` ("Pre-registered selection protocol
for Phase 3"), which was committed before the ensemble search was run.

## The model

Equal-weight **rank average** of five out-of-fold members (each member is 5
checkpoints, one per official fold; every row is scored only by the checkpoint that
held its fold out):

| # | member | checkpoints | architecture |
|---|---|---|---|
| 1 | `r4p2_ordSSM` | `checkpoints/r4p2_ordSSM_cv{1..5}/best.pt` | ordinal head + bidirectional S4D mixer |
| 2 | `r4p2_ssm` | `checkpoints/r4p2_ssm_cv{1..5}/best.pt` | simplex head + bidirectional S4D mixer |
| 3 | `r4p2_ordC` | `checkpoints/r4p2_ordC_cv{1..5}/best.pt` | ordinal head + Family-C feature branch |
| 4 | `r4p2_ordA` | `checkpoints/r4p2_ordA_cv{1..5}/best.pt` | ordinal head + layerwise FiLM |
| 5 | `r3_familyA` | `checkpoints/r3_familyA_cv{1..5}/best.pt` | simplex head + layerwise FiLM (round-3 incumbent) |

Combination rule is the frozen one from round 3: equal weights, rank average. Fitted
weights scored 0.9156 vs 0.9149 on dev — **not adopted**; round 3 established they are
unstable, the margin is below noise, and fitted weights are not part of the
pre-registered rule.

## How this was selected

| subset | k | dev (3 folds) | lockbox (OOF) |
|---|---:|---:|---:|
| ordSSM+ssm+ordC+ordA | 4 | **0.914853** | 0.914336 |
| **ordSSM+ssm+ordC+ordA+familyA** | **5** | 0.914117 | **0.915347** |
| ordSSM+ssm+ordC+ordA+ordB | 5 | 0.914200 | 0.914600 |
| ordSSM+ssm+ordA | 3 | 0.914100 | 0.913500 |
| ordSSM+ssm+ordC+ordA+familyC | 5 | 0.914000 | 0.913500 |
| *round-3 ensemble (incumbent)* | *3* | *0.8982* | *0.9003* |

The dev winner (k=4) and the lockbox winner (k=5) disagreed. The pre-registered rule
is: prefer the lockbox winner, unless the two are within 0.001 on the lockbox, in
which case prefer fewer members.

**Measured gap: 0.001011.** That is above the threshold by 0.000011, so the rule
selects the 5-member subset. This is uncomfortably marginal and it is worth stating
plainly: the two candidates are statistically indistinguishable — the k=4 subset is
better on dev by 0.00074 and worse on the lockbox by 0.00101. Either would be a
defensible choice, and the decision is low-stakes precisely because they are so
close. The rule is being followed as written rather than re-argued after seeing the
numbers, which is the entire reason it was fixed in advance.

Every subset in the table beat the round-3 ensemble on **all three** dev folds
(the unanimity requirement), not merely on the mean.

The unconstrained lockbox-best subset (0.9155, ordSSM+ssm+ordA+ordB+familyA) was
**not** considered. Selecting on the lockbox would convert it from a gate into a
selection surface and destroy the only unworn evaluation data left.

## Provenance of the numbers above

- **Dev**: three Liu+Kim-matched dev folds, out-of-fold predictions.
- **Lockbox**: 17,975 rows / 367 protospacers, scored OOF. Verified at build time to
  contain no held-out rows and to share no record and no protospacer with any dev
  validation set. OOF scoring is mandatory here because lockbox rows sit in official
  folds 1-5, so every member trained on ~4/5 of them.
- Neither surface touches the official 20,509-row held-out set.

## What happens next, committed in advance

One evaluation of this frozen ensemble on the official held-out set, with a
5000-resample protospacer-clustered paired bootstrap against OptiPrime, and the
round-3 ensemble scored on the same rows in the same run.

Whatever that returns is the reported result — **including a miss of the 0.90
target**. The search will not be re-opened against the held-out set.

## Expectation, recorded before the result is known

Dev is 0.9149 and lockbox 0.9153. Round 3's dev→held-out transfer lost ~0.005
(0.8982 dev → 0.8933 held-out). A comparable loss here implies roughly **0.910**, and
the held-out set is Liu+Kim-heavy in a way that has historically scored lower than
dev. So 0.90 looks likely to be cleared, but the point estimate could plausibly land
anywhere in 0.905-0.915, and dev→held-out transfer has surprised this project before
(round 2's dev gain reversed on held-out).
