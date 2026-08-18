# Round-2 Final Model Spec (frozen)

Per `claude_code_round2_pe_rankformer_model_search.md` §27: this document freezes
the round-2 winning model before any held-out evaluation. Nothing below should
change after this commit without re-opening model search.

## Winning recipe: Family C (feature-augmented branch)

Stage A screened 5 architectural/loss candidates against the round-1 baseline on
a fixed development split (val_fold=1, train_folds=[2,3,4,5], seed 20260812, 30
epochs). Stage B confirmed the Stage-A winner across all 5 official CV folds.

### Stage A (development split only)

| Model | val Spearman | Δ vs baseline |
|---|---:|---:|
| **Family C — feature branch** | **0.9220** | **+0.0028** |
| Family A — layerwise context | 0.9214 | +0.0022 |
| Family D — A+C combined | 0.9202 | +0.0010 |
| Baseline (round 1) | 0.9192 | -- |
| MoE4-on-layerwise (A+B) | 0.9181 | -0.0011 |
| β=0.025 correlation loss | 0.9158 | -0.0034 |

Two combination attempts (A+C, A+B/MoE) both underperformed their better single
component -- the gains from layerwise context and the feature branch overlap
rather than stack, and MoE added capacity without adding usable information.
Neither is carried forward.

### Stage B: full 5-fold CV confirmation of Family C

| Fold (val_fold) | Run | best epoch | val Spearman |
|---|---|---:|---:|
| 1 | `r2_familyC_features` | 22 | 0.9220 |
| 2 | `r2_familyC_cv2` | 26 | 0.9195 |
| 3 | `r2_familyC_cv3` | 26 | 0.9203 |
| 4 | `r2_familyC_cv4` | 21 | 0.9190 |
| 5 | `r2_familyC_cv5` | 27 | 0.9202 |

**Mean 0.9202, std 0.0011.** The gain is stable across all 5 folds, not a
fold-1 fluke -- confirms Family C as the round-2 winner.

## Frozen configuration

- **Architecture**: `context_strategy="late"` (round-1 architecture, single FiLM
  after pooling -- layerwise context was tried and did not stack with the
  feature branch, so it is off). `outcome_head="simplex"`. `moe_experts=0`.
- **Feature branch**: `n_features=16`, `feature_hidden_dim=64`. Features from
  `scripts/data/compute_family_c_features.py`: PBS/RTT length, PBS/RTT/extension
  GC%, edit length/position/position-from-nick/mismatch-count, PBS/RTT melting
  temperature, 4 ViennaRNA MFE features, RuleSet3 on-target score. Normalized
  and NaN-imputed from **training-split rows only** per fold (§9), with a
  parallel missingness mask fed into the branch.
- **Loss**: `lambda_rank=0.0`, `beta_corr=0.0` (both off -- neither ranking loss
  nor correlation loss beat the plain simplex-supervision baseline in this or
  round 1's search).
- **Training**: `configs/round2/baseline_round1.yaml` base config
  (d_model=384, 6 heads, FFN 1536, dropout 0.10, 6 edit layers, 4 pegRNA layers,
  2 cross-attention blocks, batch 512, AdamW lr=3e-4, 30-epoch budget, patience
  30), `--simplex-head --lambda-rank 0 --feature-branch`, seed 20260812.
- **Params**: 26.67M per member (26.64M base + ~34K feature branch).

## Ensemble members (frozen)

5-model CV ensemble, one member per official fold, each excluding its own
validation fold from training (collectively covering all 297,962 training rows,
matching OptiPrime's own 5-checkpoint structure and round 1's methodology):

```
checkpoints/r2_familyC_features_1787078110/best.pt   (val_fold=1)
checkpoints/r2_familyC_cv2_1787089787/best.pt         (val_fold=2)
checkpoints/r2_familyC_cv3_1787092003/best.pt         (val_fold=3)
checkpoints/r2_familyC_cv4_1787094194/best.pt         (val_fold=4)
checkpoints/r2_familyC_cv5_1787091012/best.pt         (val_fold=5)
```

Ensemble combination: unweighted mean of predicted efficiency, matching
`evaluate_ensemble.py` and round 1's methodology (no weight optimization --
per-fold OOF Spearman is too close, std 0.0011, to justify anything beyond
equal weighting; per spec §20 "optimize ensemble composition/weights on CV
predictions only" was satisfied by confirming equal weighting is appropriate
given how tightly the 5 folds agree, rather than fitting weights that would
risk overfitting to CV noise this small).

## What has NOT been done yet

The 20,509-row official held-out test set (fold 0) has **not** been queried for
this model. Per spec §2/§27/§36, that requires `--allow-heldout-evaluation` on
`evaluate_ensemble.py`, which is intentionally held back until this spec is
committed (this document) -- satisfied as of this commit -- and until a human
has had visibility into the decision to spend that query, since the held-out
set has been treated throughout this project as a single, precious resource
(round 1's own report discloses exactly how many times it was queried).

**Next step (§28, pending go-ahead):** evaluate the frozen 5-model Family-C
ensemble on the full 20,509-row held-out set, the 9,175-row Liu partition, and
the 11,334-row Kim partition; paired protospacer-clustered bootstrap against
both the round-1 ensemble and OptiPrime's official weights; report per spec
§28-29.
