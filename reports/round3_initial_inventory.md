# Round-3 Initial Inventory

Per `claude_code_round3_pe_rankformer_experiments.md` §3. Repo state at start:
`git rev-parse HEAD` = `8aa7017` (round-2 final).

## Key commits

- **Round-1 baseline**: `04a6dbc`, tagged `round1-final`. 5-model official-CV
  ensemble, held-out Spearman 0.8865 (Liu 0.8349, Kim 0.7751).
- **Round-2 final**: `8aa7017`. Family C (feature branch) 5-model ensemble,
  held-out Spearman 0.8831 (Liu 0.8298, Kim 0.7727) -- confirmed *worse* than
  round-1 on held-out despite winning validation (0.9220 vs 0.9192 mean CV
  Spearman), not statistically significant either way (p=0.18, CI includes
  zero but leans negative). Diagnosed as validation-composition mismatch:
  Schwank is 58% of every official CV fold's training/validation pool but 0%
  of the held-out set.

## Training corpus and hashes

- `data/processed/optiprime_official_318471.parquet` /
  `featurized_official.npz` (hash `2e475a6a0cdc...`, unchanged since round 2).
- 297,962 training rows (folds 1-5) + 20,509 held-out rows (fold 0).
- Composition of the 297,962 training rows: Schwank (`pridict_pridict2`)
  174,067 (58.4%), Liu (`hsu2026`) 65,594 (22.0%), Kim (`deepprime`) 58,301
  (19.6%).
- Composition of the 20,509 held-out rows: Liu 9,175 (44.7%), Kim 11,334
  (55.3%), **Schwank 0 (0%)** -- confirmed directly from the corpus, matches
  the spec's stated 44.7%/55.3% exactly.

## Official CV split definitions

Protospacer-disjoint, `src/pe_rankformer/data/folds.py::assign_folds`, seed
20260812, 5 folds built once over the *entire* 297,962+20,509-row corpus
combined (fold 0 = held-out, 1-5 = official CV). Unchanged since round 1.

## Current architecture (round-2 final state)

`src/pe_rankformer/models/pe_rankformer.py`: dual-encoder Transformer +
cross-attention, 26.6M params. Config flags added across rounds 1-2:
`context_strategy` (`"late"`|`"layerwise"`), `outcome_head`
(`"scalar"`|`"simplex"`), `n_features`/`feature_hidden_dim` (round-2 Family C
feature branch), `moe_experts` (round-2 Family B, confirmed negative, not to
be revisited per spec §18). Round-3 spec §3 says "identify... current
architecture" without mandating changes yet -- no architecture change in this
commit.

## Simplex head

Unchanged since round 1. 3-way softmax over (unedited, edited, indel);
marginalises unobserved indel (42.5% of the corpus, all Kim rows + Schwank
diverse libraries) rather than imputing zero.

## Context features

7 categorical fields (`cell_type, pe_type, cas9_type, cas9_pam, scaffold_name,
motif, source_study`) via FiLM, `context_strategy="late"` in the frozen
baseline (layerwise conditioning was round-2 Family A -- helped old validation,
never reached a held-out evaluation since it wasn't the Stage-A winner; spec
§11 asks to re-evaluate it under the corrected benchmark).

## Training/evaluation scripts

- `scripts/train/train_pilot.py` -- extended this commit with
  `--dev-folds-file`/`--dev-fold-col` (round-3 §5) to allow training/evaluating
  against the new Liu+Kim-matched development folds instead of the official
  val_fold/train_folds. Smoke-tested (1 epoch, clean run, artifacts discarded).
- `scripts/evaluate/evaluate_ensemble.py`,
  `scripts/evaluate/evaluate_familyC_heldout.py` (round-2, feature-branch
  aware) -- both gated by `--allow-heldout-evaluation` /
  `src/pe_rankformer/evaluation/heldout_guard.py`, unchanged.
- New this commit: `scripts/data/build_round3_dev_folds.py` (§5).

## Existing checkpoints usable this round

| Model | Checkpoints | Known held-out Spearman? |
|---|---|---|
| Round-1 baseline | `checkpoints/cv{1..5}_simplex_*/best.pt` (5-model ensemble) | **Yes: 0.8865** |
| Round-2 Family A (layerwise context) | `checkpoints/r2_familyA_layerwise_*/best.pt` (single fold, val_fold=1 only) | No -- Stage-A only, never reached Stage B/held-out |
| Round-2 Family C (feature branch) | `checkpoints/r2_familyC_{features,cv2,cv3,cv4,cv5}_*/best.pt` (5-model ensemble) | **Yes: 0.8831** |
| Round-2 Family D (layerwise + features) | `checkpoints/r2_familyD_layerwise_features_*/best.pt` (single fold, val_fold=1 only) | No -- Stage-A only |

Only 2 of the 4 models spec §6 asks to re-score have a true held-out number to
check the new dev-fold ranking against. This is disclosed up front rather than
implied away: §6's re-scoring exercise can confirm/deny whether the new
Liu+Kim-matched validation correctly reorders round-1 vs. Family C (it should
rank round-1 above Family C, since that's the known held-out ordering), but
cannot make the same check for Family A or Family D, which were never
evaluated on held-out data in round 2 and won't be now either (per the
absolute held-out rule, §2) until/unless they become round-3 finalists.

## OptiPrime reproduction code

Unchanged since round 1: `external/optiprime/` (real released code + weights),
`scripts/evaluate/precompute_ruleset3_cache.py` and the isolated rs3 env for
RuleSet3 features. OptiPrime's held-out predictions already computed and
cached (`results/heldout_full_head_to_head.parquet`), reused for round-3
comparisons rather than re-run.

## Compute environment

3 large GPUs currently free: RTX PRO 6000 Blackwell (97GB, index 6), 2x L40
(46GB, indices 2/7). 5x RTX 2080 Ti (11GB) available but too small for batch
size 512 without changing the variable under test; not used for round-3
screening runs, same policy as round 2.

## Frozen this commit

`configs/round3/baseline_round1.yaml` -- byte-identical model/loss/optim/train
sections to `configs/round2/baseline_round1.yaml`, with round-3-specific
header documentation. Not modified from round-1's actual architecture.
