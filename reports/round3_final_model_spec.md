# Round-3 Final Model Spec (frozen)

Per `claude_code_round3_pe_rankformer_experiments.md` §26: freezes the round-3
model before any held-out evaluation. Nothing below changes after this commit
without re-opening model search.

## What changed from the spec's plan, and why

The spec's stated priority order was: fix validation → domain adaptation →
source specialization → context → scaling → pretraining → **ensemble (last)**.
Round 3 instead promoted ensembling to the primary strategy after Stage 0 showed
that single-model effects at this project's scale (~0.001–0.004 Spearman) are
mostly unresolvable by any evaluation set available (dev or held-out, ~750–800
protospacer clusters), while ensemble-diversity effects are an order of
magnitude larger and cleanly significant. Full reasoning and every intermediate
result in `reports/round3_research_log.md`.

## Winning composition

**Family C + DAPT + Family A**, rank-average, equal weights. **Round-1 baseline
explicitly excluded** — verified to significantly *hurt* the ensemble (paired
bootstrap, p<0.0001/p<0.0001/p=0.034 across the 3 dev folds), because it is
0.997-correlated with DAPT and only dilutes the other two members' distinct
signal.

### Members (each a 5-checkpoint official-fold ensemble)

| Member | Architecture | Standalone dev Spearman | Checkpoints |
|---|---|---:|---|
| **Family C** | Round-2 feature branch (16 features incl. RuleSet3), late FiLM context | 0.8816 | `checkpoints/r2_familyC_{features,cv2,cv3,cv4,cv5}_*/best.pt` |
| **DAPT** | Round-1 architecture, fine-tuned on Liu+Kim only (lr 3e-5, 10 epochs, `--init-from` the matching round-1 official-fold checkpoint) | 0.8820 | `checkpoints/r3_dapt_lr3e5_cv{1..5}_*/best.pt` |
| **Family A** | Layerwise FiLM context conditioning (every block, not just after pooling), trained from scratch, no feature branch | 0.8845 | `checkpoints/r3_familyA_cv{1..5}_*/best.pt` |

Every member's 5 checkpoints follow the official protospacer-disjoint fold
structure (one checkpoint per official fold 1–5, trained on the other four),
collectively covering all 297,962 training rows, matching OptiPrime's own
5-checkpoint structure.

### Why these three specifically

Rank-prediction correlation matrix (dev fold 1), the basis for every inclusion/
exclusion decision:

| | round-1 | Family C | DAPT | Family A |
|---|---:|---:|---:|---:|
| round-1 | 1.000 | 0.955 | 0.997 | 0.946 |
| Family C | 0.955 | 1.000 | 0.955 | 0.945 |
| DAPT | 0.997 | 0.955 | 1.000 | 0.947 |
| Family A | 0.946 | 0.945 | 0.947 | 1.000 |

DAPT ≈ round-1 (0.997) — fine-tuning at this LR/epoch budget barely moves the
function, so DAPT supersedes round-1 in an ensemble rather than adding to it.
Family A is the most decorrelated member from everything (0.945–0.947), which
is why it is both the strongest standalone model and the strongest ensemble
addition.

## Validated performance (matched Liu+Kim dev folds, OOF, never touching held-out)

| Configuration | Mean dev Spearman | Δ vs round-1 |
|---|---:|---:|
| round-1 baseline (single) | 0.8798 | — |
| Best single model (Family A) | 0.8845 | +0.0047 |
| Best 2-way blend (Family C + DAPT) | 0.8919 | +0.0121 |
| **Frozen ensemble (Family C + DAPT + Family A)** | **0.8982** | **+0.0184** |

Per-fold: 0.8979 / 0.8997 / 0.8969 — consistent across all 3 independently-drawn
Liu+Kim-matched development folds.

## Statistical validation performed before freezing

Two paired, protospacer-clustered bootstrap tests (2000 resamples each, on
every dev fold), specifically to rule out the frozen composition being an
artifact of searching many subsets:

1. **3-way vs. best 2-way**: +0.0069 / +0.0062 / +0.0058, all CIs exclude zero,
   100% bootstrap wins, p<0.0001 on all three folds.
2. **3-way vs. 4-way (adding round-1 back)**: round-1 significantly *hurts*
   on all three folds (p<0.0001, p<0.0001, p=0.034).

Also checked and rejected: OOF-fitted per-member ensemble weights never beat
equal weighting (0.8907 vs. 0.8909 two-member; 0.8916 vs. 0.8919 three-member on
an earlier composition), and the fitted optimum was unstable fold-to-fold.
Equal-weight rank-averaging is the frozen combination rule.

## Combination mechanics

1. Each member's 5 checkpoints predict on the held-out rows; average → one
   prediction per member (matching how OptiPrime's own 5-checkpoint release is
   evaluated).
2. Convert each member's predictions to fractional ranks in [0, 1].
3. Average the three members' ranks, equal weight.

Rank-averaging (not mean-of-efficiency) was selected because it beat mean
averaging on every dev fold tested. One consequence, disclosed rather than
worked around: the final blended output is a rank score in [0, 1], not a
calibrated probability/efficiency estimate. Spearman correlation (the primary
metric throughout this project) is invariant to this; MAE/RMSE against the
blend are not meaningful and are reported per-member only, where the scale is
genuine predicted efficiency.

## What has NOT been done yet

The 20,509-row official held-out set has not been queried for this ensemble.
Per spec §2/§26, that requires `--allow-heldout-evaluation` on
`scripts/evaluate/evaluate_heterogeneous_heldout.py`, held back until this
document is committed (satisfied as of this commit).

**Next step (§27, pending go-ahead):** evaluate the frozen ensemble on the full
20,509-row held-out set, the 9,175-row Liu partition, and the 11,334-row Kim
partition; paired bootstrap against OptiPrime, round-1, and round-2; per-
condition breakdown and count of Kim conditions won.
