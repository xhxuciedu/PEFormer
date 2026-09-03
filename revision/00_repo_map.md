# Repo map — artifacts the revision plan depends on

Manuscript: **`reports/paper/pe_rankformer_paper.tex`** (the `<PATH TO MANUSCRIPT>`
placeholder in the brief was not filled in; this is the only manuscript in the repo).
Compiled PDF alongside it. Figures are read from `results/paper/figures/`.

All paths below verified to exist. Anything the plan assumes and I could **not** find is
in §Gaps.

## Corpus and splits

| what | path | detail |
|---|---|---|
| Full corpus | `data/processed/optiprime_official_318471.parquet` | 318,471 rows × 29 cols. `fold==0` is the 20,509-row held-out set; `fold∈1..5` are the 297,962 training rows |
| Model input tensors | `data/processed/featurized_official.npz` | `edit_ids` (N,102), `peg_nuc_ids` (N,90), `peg_seg_ids`, `target`, `target_indel`, `group_key`, `fold`, `record_id`, 7 `ctx_*` arrays. **No `weight` column** — read it from the parquet by `record_id` |
| Protospacer ID | `spacer` column of the corpus | **750 distinct values on the held-out set.** This is the clustering unit for every bootstrap |
| Official CV folds | `fold` column of the corpus | protospacer-disjoint |
| Matched dev folds | `data/processed/round3_dev_assignments.parquet` | `record_id` + `round3_dev_fold_{0,1,2}`, string-valued `'train'`/`'val'`. Reported dev numbers are the `'val'` rows (~35.7k each) |
| Lockbox | `data/processed/round4_lockbox.parquet`, `round4_lockbox_folds.parquet`, `round5_lockbox_folds.parquet` | 17,975 rows / 367 protospacers |
| Engineered features | `data/processed/family_c_features.parquet` | 17 features: PBS/RTT length + GC, 4 Tm, 4 ViennaRNA MFE, edit type/length/offset, mismatch count, RuleSet3 score |
| Context vocab | `data/processed/context_vocab_official.json` | the 7 fields the model sees, incl. `source_study`. **43 distinct context cells in total** |
| Authors' raw CSVs | `data/optiprime_train_mix/` (gitignored) | 58 files; carry the per-row `weight` column and, for Liu, `mmr_weight` |

## Frozen predictions — read only

| what | path | detail |
|---|---|---|
| Held-out, frozen system | `results/round4/heldout/predictions_round4_final.parquet` | 20,509 rows; ensemble `predicted_efficiency` **plus per-member columns** `member_{ordSSM,ssm,ordC,ordA,familyA}` |
| Same + calibration | `results/round5/heldout_calibrated.parquet` | adds `calibrated_efficiency` (isotonic). Used by most round-9 scripts |
| Head-to-head vs OptiPrime | `results/heldout_full_head_to_head.parquet` | `y`, `op` (OptiPrime), and `predicted_efficiency` — **note: that column is the round-1 model (0.8865), not the final one.** Join the final from the two files above |
| OptiPrime, Liu only | `results/optiprime_heldout_predictions.parquet` | 9,175 rows |
| Dev OOF, per member | `results/round3/dev_recalibration/predictions_<run>_round3_dev_fold_{0,1,2}.parquet` | 107 files, one per (run × fold) |
| Round-3 held-out | `results/round3/heldout/predictions_r3_final_ensemble.parquet` | earlier generation |
| Checkpoints | `checkpoints/<run_id>/{best,final}.pt` | 156 with `best.pt`. `model_config` is a **dict**, so read thresholds with `.get()`, not `getattr` |
| Run metadata | `results/runs/<run_id>/{run_info.json,config.yaml,training_history.csv}` | 181 runs; git commit, dataset SHA, best epoch, per-epoch val Spearman |
| Held-out access log | `logs/heldout_evaluations.log` | 3 gated accesses + round 1 = 4 total |

## Analysis and figure scripts

| what | path |
|---|---|
| Paper figures (6) | `scripts/evaluate/make_paper_figures.py` — `fig1_architecture` … `fig6_ceiling` |
| Replicate / noise ceiling | `scripts/evaluate/noise_ceiling.py` (Gaussian, superseded), `noise_ceiling_empirical.py` (fold 0 only), `noise_ceiling_surfaces.py` (round 9, all 5 surfaces) |
| Paired bootstrap | `scripts/evaluate/paired_bootstrap_test.py`, `round4_final_bootstrap.py` |
| Ensemble search / diversity | `search_ensemble.py`, `diversity_report.py`; results in `results/round4/diversity_dev0.json`, `diversity_wave2.json` |
| Tie floor | `scripts/evaluate/tie_floor.py`, `results/round5/tie_floor.json` |
| Factorial | `scripts/evaluate/factorial_study.py`, `results/round5/factorial.json` |
| Training | `scripts/train/train_pilot.py` (all mechanisms behind flags) |
| Round-9 additions | `stratified_comparison.py`, `gbm_baseline.py`, `early_stopping_survey.py`, `noise_ceiling_surfaces.py`; results in `results/round9/` |

## Replicate-group analysis

Defined in `noise_ceiling_empirical.py`: Kim rows grouped on **17 keys** (`FULL_KEY`) —
16 design/condition covariates plus `target_name`. 649 groups, 1,298 ordered pairs
(1,162 after the symmetrisation used in round 9). Used for the ceiling and for the
censoring quantiles.

## Gaps — things the plan assumes that are absent

1. **Cross-attention maps are not extractable** (Phase 2.3). Every attention call passes
   `need_weights=False` (`pe_rankformer.py:255,256,277`). Visualising PBS/RTT→target
   alignment needs a code change to return and plumb out the maps. Feasible, not free.
2. **No relative-position attention and no dilated-convolution mixer** (Phase 2.2).
   `--sequence-mixer` offers `attention | ssm | hybrid_alt | hybrid_par | selective |
   selective_frozen`. ALiBi/T5 bias and a depthwise dilated stack must be implemented.
3. **No MMR status annotation and no chromatin scores** (Phase 2.4). Neither exists in
   the repo in any form — only prose mentions in reports. Cell-line MMR status must be
   curated from primary literature (the brief is right to insist; I have not encoded it
   from recall) and ePRIDICT scores obtained externally.

   **A design caveat worth settling before this costs GPU time.** MMR status is a
   deterministic function of the experimental condition here, and the condition is
   already an input. The corpus's own `MLH1dn` flag is fully determined by the seven
   context fields the model receives (measured: 0 rows in which it varies within a
   context cell, across all 43 cells) — and `pe_type` encodes PE2 vs PE4, where PE4
   *is* PE2 + MLH1dn. Cell-line *intrinsic* MMR status would likewise be a function of
   `cell_type`, which is also already an input.

   So adding MMR status as a feature cannot add **information** to a model that already
   sees cell line and PE system; it can only add **inductive bias** — telling the model
   which cell lines should behave alike. That predicts no gain in the current
   within-corpus split, and it is consistent with the round-8 per-source-head result
   (−0.0018) and the round-6/7 context experiments all coming back null.
   The experiment where the bias can pay is **leave-one-cell-line-out**: there the model
   must score a line it has never seen, a bare categorical embedding is uninformative for
   it, and MMR status is the bridge. I would run Phase 2.4 in that design, not as a
   feature addition to the existing split, and I would state the prediction first.
4. **`weight` is absent from the featurized npz** — join from the parquet (already done
   by `train_pilot.py --row-weights`).
5. **No leave-one-study-out results yet** (Phase 2.5), but `--train-sources` and
   `--val-sources` already exist, so this needs no new code.

## Correction to a manuscript limitation

The draft states PRIDICT2.0 and DeepPrime-FT could not be run because they need "a
separate pretrained on-target cutting model". **Those weights are present in the repo:**

- `external/pridict2/trained_models/DeepCas9_Final/` — TensorFlow checkpoint (2.8 MB)
- `external/pridict/DeepCas9_Final/` — second copy
- `external/pridict2/trained_models/pridict1_1`, `pridict1_2` — PRIDICT2 weights
- `external/deepprime/models/ontarget_variants/DP_variant_*` — per-condition DeepPrime-FT
  checkpoints, including the exact ones the DeepPrime paper's Fig. 4 specifies

What actually remains is reproducing the Tm and ViennaRNA MFE feature computations
byte-compatibly with each tool's own preprocessing. That is a real obstacle but a
narrower one than the manuscript claims, and the limitation should be reworded whether
or not Phase 2.6 completes.
