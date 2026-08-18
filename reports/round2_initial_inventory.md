# Round-2 Initial Inventory

Per `claude_code_round2_pe_rankformer_model_search.md` §3. Repo state at
`git rev-parse HEAD` = `04a6dbc` (tagged `round1-final`).

## Current architecture (`src/pe_rankformer/models/pe_rankformer.py`)

26.6M-parameter dual-encoder Transformer with cross-attention and late FiLM
context conditioning:

- **Edit encoder**: 6-layer pre-LN Transformer encoder over a paired
  unedited/edited token sequence (`EDIT_MAX_LEN=100` + BOS/EOS), `d_model=384`,
  6 heads, FFN 1536.
- **pegRNA encoder**: 4-layer encoder over spacer+PBS+RTT tokens
  (`PEG_MAX_LEN=90`) with segment-type embeddings distinguishing the three
  regions.
- **Cross-attention**: 2 bidirectional blocks (edit attends to pegRNA and vice
  versa), each with its own post-attention FFN.
- **Pooling**: one learned-query attention pool per stream, concatenated to a
  768-dim vector.
- **Context conditioning**: applied **once**, after pooling, via a single
  FiLM layer: `context_vector -> (gamma, beta)`, `h' = (1+gamma)*h + beta`.
  This is the target of Family A (§7) — conditioning is late and coarse, not
  per-block.
- **Context fields** (`src/pe_rankformer/data/context.py`): `cell_type,
  pe_type, cas9_type, cas9_pam, scaffold_name, motif, source_study` — 7
  categorical fields, each with its own embedding (`context_embed_dim=32`),
  concatenated to a 224-dim context vector before the FiLM MLP.
- **Head**: `LayerNorm -> Linear(768,384) -> GELU -> Dropout -> Linear(384, n_out)`,
  `n_out=1` (scalar) or `n_out=3` (simplex).

Total: 26,642,851 params (simplex head; 26,642,081 for scalar, per
`run_info.json` across round-1 runs).

## Current losses (`src/pe_rankformer/training/losses.py`)

- **Regression**: Huber, either raw `[0,1]` space or clipped-logit space
  (`regression_space` config key). Round-1 found these statistically
  equivalent; raw is the default.
- **Ranking**: pairwise RankNet (`softplus(-(s_i-s_j))`) over within-target
  pairs sampled by `training/ranking.py::sample_ranking_pairs`
  (`min_pair_diff=0.02`, `max_pairs_per_group=4`). Round-1 found
  `lambda_rank>0` **costs** global validation Spearman at every weight tried
  (0.05, 0.25) — the round-1 winning config uses `lambda_rank=0`. This is the
  objective §13 of the round-2 spec asks to revisit with a correlation-aware
  loss instead.
- **Simplex head loss**: soft cross-entropy over `(unedited, edited, indel)`
  proportions; marginalises the indel class for the 42.5% of the official
  corpus with unmeasured indel (all Kim rows, Schwank diverse libraries)
  rather than imputing zero. This was the single largest round-1 architectural
  win.

## Context representation

Purely categorical, 7 fields, single late FiLM injection (see above). No
continuous features (PBS/RTT length, GC content, MFE, RuleSet3/DeepSpCas9
activity, etc.) are used anywhere in the model — this is exactly the gap
Family C (§9) targets.

## Simplex head implementation

`PERankFormerConfig.outcome_head: "scalar"|"simplex"`. Simplex path: 3-way
softmax head; `ranking_score()` returns `logit_edited - logit_unedited` (the
log-odds of correct edit vs. no edit, invariant to the indel logit) for use in
the ranking loss; `efficiency_from_output()` returns `softmax(out)[:,1]`.
`simplex_loss()` handles missing indel by marginalisation (see above), with 3
regression tests covering: minimization at true proportions, `edited+indel>1`
clamping, marginalisation vs. imputation divergence, and gradient finiteness
under all-NaN indel.

## Train/validation/CV protocol

- **Corpus**: `data/processed/optiprime_official_318471.parquet` /
  `featurized_official.npz` — OptiPrime's authors' own training mix.
  318,471 rows total: 297,962 training (folds 1–5, OptiPrime's own CV splits)
  + 20,509 held-out test (fold 0).
- **Round-1 winning protocol**: 5-model CV ensemble, model *k* trained with
  `val_fold=k`, `train_folds`=the other four members of `{1..5}`. Each
  ensemble member therefore trains on 4/5 of the 297,962 rows; the 5-model
  ensemble collectively covers all of it, matching OptiPrime's own released
  5-checkpoint structure.
- **Batching**: `GroupedBatchSampler` biases batches toward shared
  within-target ranking groups (`training/ranking.py`), needed because
  ~220k+ distinct groups make uniform batching yield ~0 rankable pairs/batch.
- **Optimization**: AdamW (`lr=3e-4`, `wd=0.01`), warmup+cosine schedule,
  BF16 autocast, grad clip 1.0, batch size 512, up to 30 epochs,
  early-stop patience 30 (round-1 found patience=5 unfairly penalized the
  no-context ablation; 30 is used for all final-sweep and official-corpus runs).

## Current best validation performance (per-fold, round-1 winning config)

| val_fold | best epoch | best val Spearman |
|---|---|---|
| 1 | 22 | 0.9192 |
| 2 | 24 | 0.9173 |
| 3 | 24 | 0.9161 |
| 4 | 26 | 0.9174 |
| 5 | 29 | 0.9202 |

(checkpoints: `checkpoints/cv{1..5}_simplex_*/best.pt`)

## Known round-1 failure modes

1. **Ranking loss hurts global Spearman.** `lambda_rank>0` costs ~0.02-0.08
   validation Spearman at every weight tried (0.05, 0.25), vs. `lambda_rank=0`.
   Round-2 §13 explicitly forbids reusing `lambda_rank=0.25` and asks for a
   correlation-aware loss instead.
2. **Padding artifact retracted a "decisive win."** An early result comparing
   PE-RankFormer against OptiPrime on a public-data reconstruction used
   filler-padded sequences for OptiPrime's input, depressing its score by
   ~0.11 Spearman — 70× the real effect. Retracted once the authors' official
   data made a padding-free comparison possible. Lesson institutionalised
   in the held-out guard added this session (see §Guard below) and in
   `reports/pilot_results.md` §0.5.
3. **Reproduction ceiling vs. the paper.** Our OptiPrime reproduction scores
   ~0.03 Spearman *above* the paper's own published Liu-only figure
   (0.805 vs. 0.775) under every aggregation convention checked. This bounds
   how finely any head-to-head claim can be trusted; round-2 comparisons
   against OptiPrime should keep this in mind but it does not block model
   *search*, since search happens on our own dev/CV metrics, not against
   OptiPrime.
4. **Ensemble-size / training-coverage asymmetry.** An intermediate run using
   3 seeds on 80% of the training data scored 0.0120 Spearman worse than the
   properly matched 5-fold protocol — comparable in size to the model
   differences under study. Round-2 must keep ensemble composition and
   training coverage matched across every comparison (Stage A: same data/
   split/seed/epochs/eval code per §6; Stage B: full 5-fold for finalists only).
5. **Indel supervision is uneven and untested on the official corpus.** The
   simplex head's ablation gain was measured on the earlier public-data
   reconstruction (complete indel coverage); it has not been re-ablated on
   the official corpus, where 42.5% of rows have no measured indel.

## Current held-out (test-fold) results — DO NOT re-touch until round-2 freeze

| Scope | n | PE-RankFormer ρ | OptiPrime ρ | Δρ |
|---|---:|---:|---:|---:|
| Full held-out | 20,509 | 0.8865 | 0.8690 | +0.0175 (CI [+0.007,+0.028], p=0.001) |
| Liu only | 9,175 | 0.8349 | 0.8365 | −0.0016 (p=0.94, dead heat) |
| Kim only | 11,334 | 0.7751 | 0.7320 | +0.0431 |

Full detail: `reports/pilot_results.md` §0, `reports/pilot_results.pdf`.

## Compute profile

- Hardware: single RTX PRO 6000 Blackwell (97GB), currently free; GPUs 2/7
  (46GB) also free.
- One 30-epoch run at the current architecture/corpus size: ~25–36 min wall
  clock (round-1 logs: 27–36 min per member across the official-corpus runs),
  peak VRAM well under capacity at batch size 512.
- A 5-fold CV sweep of one config: ~2–3 GPU-hours.

## Guard added this session (§2, §36)

`src/pe_rankformer/evaluation/heldout_guard.py::require_heldout_permission`.
Wired into both scripts capable of scoring the held-out fold:

- `scripts/evaluate/evaluate_ensemble.py` — `--split` default changed from
  `test` to `val`; `--split test` now additionally requires
  `--allow-heldout-evaluation`.
- `scripts/evaluate/evaluate_pe_rankformer.py` — always evaluated the test
  fold unconditionally before this change; now requires
  `--allow-heldout-evaluation` unconditionally.

Every permitted access is appended as a timestamped JSON record to
`logs/heldout_evaluations.log` (script, reason, argv, user, UTC timestamp).
Refusal raises `SystemExit` with an explanation pointing at this document.
