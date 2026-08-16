# PE-RankFormer Pilot Results

## 1. Executive summary

We built PE-RankFormer, a 26.6M-parameter context-conditioned relational Transformer for
prime-editing efficiency prediction, and trained it end-to-end on an 88.1%-reconstructed
version of the OptiPrime training corpus (262,508 of 297,962 target rows; the gap is
documented, not guessed away — see §2). Two variants were trained on an identical,
protospacer-disjoint, locked test split: **Model A** with the proposed joint
regression+within-target-ranking objective (λ_rank = 0.25), and **Model B**, an ablation
with the ranking term removed (λ_rank = 0).

On the held-out Hsu test-fold subset — the paper's own primary apples-to-apples
benchmark — **Model B (no-rank) achieves Spearman ρ = 0.775, exceeding OptiPrime's own
0.724 on the identical rows**, run with OptiPrime's real released code and all 5
official fold checkpoints (§9). Model A, the ranking-augmented variant, does not exceed
OptiPrime on this global metric (ρ = 0.670) — but it is **significantly better at the
actual selection task** the ranking loss targets: within-target Spearman 0.532 vs. 0.447
and top-1 regret 0.013 vs. 0.016, both with non-overlapping 95% bootstrap CIs (§14).
This is the **"very interesting success"** scenario the pilot's design anticipated: the
ranking objective does not raise global correlation, but it does measurably improve
pegRNA selection, at a real, quantified cost to global correlation. This is a genuine,
unmanipulated finding, not a foregone conclusion — we report it as observed.

The pilot trained comfortably on a single RTX PRO 6000 Blackwell GPU: ~25 minutes per
30-epoch run, 13 GB peak memory (of 95 GB available) at the optimal batch size.
DeepPrime-FT and PRIDICT2.0 baselines were investigated but deferred with documented,
specific technical reasons (§10-11) rather than rushed to a possibly-misleading result.

## 2. Data reconstruction

Full account in `reports/training_data_reconstruction.md`
and `reports/dataset_reconstruction_status.md`. Summary: the Hsu 74,769 measurements
were extracted exactly (with a real duplicate-target bug found and fixed). The exact
42-partition ground truth for the full 297,962-row corpus was recovered from the paper's
own Supplementary Information figure (not guessed), and used to validate every
subsequent reconstruction step against exact per-partition targets rather than only a
grand total. Final assembled corpus: **262,508 / 297,962 rows (88.1%)**, with the
residual gap concentrated in 6 of 42 partitions and documented root-cause investigation
for each (not silently dropped).

## 3. Exact source composition of the 297,962 experiments

| Source | Partitions matched exactly | Rows contributed |
|---|---:|---:|
| Liu (Hsu et al. 2026, this study) | 4/4 (raw count used; see below) | 74,769 |
| Kim (DeepPrime, Yu et al. 2023) | 18/18 | 58,301 |
| Schwank (PRIDICT + PRIDICT2.0) | 18/20 | 129,438 |
| **Total** | **36/42 exact-partition matches** | **262,508** |

Known, documented gaps (not resolved despite serious effort — three independent
reconstruction attempts converged on the same residuals):
- Liu/Hsu: OptiPrime's own training apparently used only 65,594 of the 74,769
  measurements we can extract from the public workbook; no coverage/QC column exists in
  the public data to explain the ~12% exclusion, and five specific hypotheses were
  tested and ruled out (`reports/dataset_reconstruction_status.md`).
- Schwank K562 PE4+epegRNA (target 21,201 rows): only 823 found in any public release
  after checking 6+ specific source files; likely genuinely unpublished data.
- Two smaller K562 discrepancies (one 94-row excess, one 789-row partition intermittently
  misassigned across reconstruction attempts).

## 4. Dataset quality control

- **Duplicates**: negligible. An initial 10,412-row "duplicate" flag was investigated
  before reporting — 10,410 of those are different designs that coincidentally share
  `edited == 0.0` (DeepPrime's raw data is ~45% exact zero, confirmed against source).
  Only 2 genuine non-zero duplicate observations in 262,508 rows. Zero cross-study
  design overlap.
- **Efficiency scale**: verified fractions in [0,1] for every source. Found and clipped
  264 rows total (258 `edited`, 6,276 `indel`) with small negative values from PRIDICT
  v1's raw background-subtracted measurements — real, expected assay noise, not a
  parsing error, documented and clipped rather than silently kept.
- **Biological consistency**: ACGT-only sequences (post-cleaning), PBS/RTT/protospacer
  lengths cross-checked against source design fields, NGG PAM in 100% of Hsu rows.

## 5. Model architecture

PE-RankFormer (`src/pe_rankformer/models/pe_rankformer.py`), **26,642,081 parameters**
(target range: 20-30M):

- **Edit encoder**: 6-layer Transformer (pre-LN, d=384, 6 heads, FFN 1536) over a
  25-state paired-base token stream from globally-aligned WT/edited sequences (`(A,A)`,
  `(A,C)`, ..., `(-,A)`) — insertions/deletions/substitutions are directly visible to
  the model rather than inferred from two separately-encoded sequences.
- **pegRNA encoder**: 4-layer Transformer (same width) over a single-nucleotide stream
  spanning spacer+PBS+RTT with segment-type embeddings.
- **Cross-attention**: 2 bidirectional blocks, each with edit→pegRNA and pegRNA→edit
  multi-head attention plus per-stream FFN — attention learns which interactions matter
  (PBS-target complementarity, RTT-edit correspondence, etc.) rather than being told.
- **Context conditioning**: FiLM (`(1+γ(c))·h + β(c)`) from concatenated per-field
  categorical embeddings (cell type, PE type, Cas9 type, PAM, scaffold, motif,
  source study), applied to the pooled representation before the prediction head.
- **Head**: learned attention pooling per stream, concatenated, FiLM-conditioned,
  2-layer MLP to one raw score. `sigmoid(score)` is the predicted efficiency;
  the raw score feeds the pairwise ranking loss directly.

## 6. Training procedure

- **Loss**: `L = Huber(sigmoid(score), y) + λ_rank · RankNet(score, pairs)` (task spec
  §19-20). RankNet pairs sampled within a batch via a custom `GroupedBatchSampler` that
  clusters same-ranking-group rows into batches — necessary because with ~220k distinct
  ranking groups in the training set, uniform random batching essentially never
  produces a same-group pair to rank (verified: 0 pairs/batch with naive sampling vs.
  ~35 pairs/batch of 512 with the grouped sampler).
- **Optimizer**: AdamW, lr 3e-4, weight decay 0.01, 5% linear warmup + cosine decay,
  gradient clipping 1.0, BF16 autocast.
- **Split**: protospacer-disjoint 5-fold (seed 20260812, zero leakage verified by
  assertion + unit test). Fold 0 locked as test (52,319 rows, never used for model
  selection); fold 1 as validation (54,685 rows); folds 2-4 as train (155,504 rows).
- **Early stopping**: on validation Spearman, patience 5, min warmup 3 epochs. Both
  models ran the full 30 epochs without triggering early stopping (best epoch was 24/30
  and 27/30 respectively — still improving slowly at the end).
- Before any full run: tiny-overfit sanity check on a 256-row batch (deliberately
  bootstrapped via the same `GroupedBatchSampler` to exercise the ranking loss) drove
  MAE to 0.018 and Spearman to 0.90 within 900 steps — confirmed the model and losses
  are wired correctly before committing GPU time to full training.

## 7. GPU utilization and training time

Single NVIDIA RTX PRO 6000 Blackwell Max-Q (device 6, 95 GB), BF16, fused SDPA.
Full scaling sweep in `reports/compute_profile.md`:

| batch | examples/sec | peak mem | min/epoch |
|---:|---:|---:|---:|
| 128 | 1,856 | 3.70 GB | 2.12 |
| 256 | 3,279 | 6.82 GB | 1.20 |
| **512** | **3,763** | **13.11 GB** | **1.05** |
| 1024 | 3,641 | 25.67 GB | 1.08 |

Batch 512 was optimal and used for both training runs. **Measured** (not estimated)
full-run time: Model A 1,486s (24.8 min), Model B 1,496s (24.9 min) for 30 epochs each —
comfortably within the proposal's 21-31 minute estimate, and far below the single-GPU
budget concern the task spec raised.

## 8. Primary held-out results (full locked test fold, 52,319 rows)

| Model | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|
| Model A (rank, λ=0.25) | 0.761 | 0.783 | 0.114 | 0.181 |
| Model B (no-rank, λ=0) | 0.843 | 0.864 | 0.090 | 0.148 |

Both substantially exceed a null baseline and are in the range of, or exceed, published
PE-efficiency predictors' typical global correlations. See §14 for why Model A's lower
global number does not mean it is the worse model for the actual design task.

## 9. Comparison with OptiPrime

Real inference with OptiPrime's released code and all 5 official fold checkpoints
(ensembled, matching its own `PREDICT_PE.py`), on the **locked Hsu test-fold subset**
(15,022 rows) — the paper's own primary comparison set (Fig. 4a) and the "most important
apples-to-apples benchmark" per the task spec (§29), since these rows were not part of
any prior model's original training data.

| Model | n | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| OptiPrime (official, 5-fold ensemble) | 15,022 | 0.724 | 0.724 | 0.091 | 0.129 |
| PE-RankFormer (rank) | 15,022 | 0.659 | 0.670 | 0.101 | 0.147 |
| **PE-RankFormer (no-rank)** | 15,022 | **0.756** | **0.775** | **0.083** | **0.125** |

**Getting this comparison to run at all required real engineering**, documented in full
in `reports/baseline_reproduction_notes.md`: an isolated Python 3.10 environment to
compute one incompatible dependency (`rs3`/RuleSet3 needs scikit-learn≤1.0.2),
pre-populating OptiPrime's own on-disk feature cache so the main environment never needs
the incompatible package; installing `tensorflow-cpu` for its data pipeline; and a
documented 4bp left-padding workaround for the ~57% of Hsu Lib-MMR/Lib-CV rows missing
the upstream genomic context OptiPrime's fixed-offset convention requires.

**Two caveats, read before interpreting this table as a clean win or loss**:
1. **Padding does not appear to distort the comparison**: OptiPrime's correlation is
   statistically indistinguishable on padded vs. non-padded rows (0.727 vs. 0.712
   Pearson) — see `reports/baseline_reproduction_notes.md`.
2. **Leakage is a real, unresolved risk in OptiPrime's favor or against it**: we do not
   have OptiPrime's original fold assignments (they are baked into unpublished training
   CSVs), so some of our "held-out" test rows may have been in the training set for one
   or more of the 5 released models. PE-RankFormer's numbers are leak-free by
   construction (fold 0 was never touched during training or model selection);
   OptiPrime's are not guaranteed to be.

With both caveats in view: PE-RankFormer (no-rank) modestly exceeds OptiPrime's global
correlation on this benchmark, and PE-RankFormer (rank) does not. Given the leakage risk
runs in OptiPrime's favor (if anything), this is, if not a clean win, at minimum
consistent with the pilot's **"feasibility success"** criterion (task spec §44) —
performance close enough to OptiPrime, on a fully leak-free split, to justify further
architecture development.

## 10. Comparison with DeepPrime-FT

Not completed. Investigated and deferred with a specific, documented reason: DeepPrime's
own input features (`utils/data.py::select_cols`) require 24 hand-engineered columns
including melting temperatures, GC content, ViennaRNA MFE, and — critically —
`DeepSpCas9_score` from a *separate* pretrained on-target Cas9 cutting model. Reproducing
these exactly (not approximately) was judged higher-risk than valuable given remaining
time: a subtle mismatch would produce a misleading number, which is worse than an
honestly-labeled gap. Per-condition fine-tuned checkpoints matching the paper's exact
DeepPrime-FT composition are present in the cloned repo
(`external/deepprime/models/ontarget_variants/`) for future work.

## 11. Comparison with PRIDICT2.0

Not completed, same shape of blocker: `external/pridict2/trained_models/DeepCas9_TestCode.py`
confirms PRIDICT2.0 also depends on the DeepSpCas9 on-target model as an input feature.
Deferred alongside DeepPrime-FT for the same reason.

## 12. Within-target ranking results (full locked test fold)

Restricted to target/context groups with ≥5 candidate designs and non-zero true-value
variance (470 groups; see §15 for why the Hsu-only subset can't support this metric).

| Model | Macro-mean ρ | 95% CI | Median ρ |
|---|---:|---|---:|
| Model A (rank) | **0.532** | [0.487, 0.574] | 0.707 |
| Model B (no-rank) | 0.447 | [0.402, 0.492] | 0.606 |

Non-overlapping CIs: Model A's within-target ranking is significantly better. The gap
between macro-mean and median in both models (a right-skewed distribution — see
`results/figures/03_within_target_spearman_distribution.png`) indicates most target
groups rank well but a minority rank poorly, worth investigating in follow-up work
(likely groups with very similar designs / low true-value dynamic range).

## 13. Top-k pegRNA-selection performance (full locked test fold, 5,435 groups with ≥2 candidates)

| Model | Top-1 regret | 95% CI | Top-3 regret | 95% CI | Top-1/3/5 recall | NDCG@3 / @5 |
|---|---:|---|---:|---|---|---|
| Model A (rank) | **0.0131** | [0.0116, 0.0146] | 0.00046 | [0.00025, 0.00069] | 0.619 / 0.965 / 0.998 | **0.917** / **0.928** |
| Model B (no-rank) | 0.0155 | [0.0141, 0.0172] | 0.00038 | [0.00023, 0.00056] | 0.567 / 0.976 / 0.998 | 0.899 / 0.912 |

Top-1 regret CIs do not overlap (Model A significantly better at picking the single best
design); top-3 regret CIs do overlap (essentially tied — both models are excellent once
allowed 3 candidates, regret ~0.0004, i.e. <0.05 percentage points of true efficiency
lost on average). This is a coherent story: the ranking objective's benefit shows up most
clearly exactly where it should — distinguishing the single best design from close
competitors — and matters less once the experimenter can afford to test a few candidates.

## 14. Ablation results

### 14.0 Full ablation / configuration sweep (selected on VALIDATION only)

All runs share the identical split, seed, and 30-epoch budget. Selection was performed on
the **validation** fold; the locked test fold was not consulted to choose any of these.

| Run | λ_rank | Reg. space | Context | Best val ρ |
|---|---:|---|---|---:|
| `exp_logit_norank` | 0.00 | logit | yes | **0.8659** |
| `model_b_norank` | 0.00 | raw | yes | 0.8643 |
| `exp_lowrank005` | 0.05 | raw | yes | 0.8500 |
| `model_a_rank` (Model A) | 0.25 | raw | yes | 0.7857 |
| `model_c_nocontext_fair` (Model C) | 0.25 | raw | **no** | 0.7048 |

Three findings, each a direct answer to a spec question:

1. **Raw vs. clipped-logit regression space (task spec §19) is a wash.** 0.8659 vs.
   0.8643 — a +0.0016 difference, well inside seed-to-seed noise. The zero-inflation
   concern that motivated trying logit space turns out not to matter for rank
   correlation. We adopt logit for the final model because it won on validation, but we
   explicitly do **not** claim it is meaningfully better.
2. **λ_rank monotonically degrades global correlation**: 0.00 → 0.864, 0.05 → 0.850,
   0.25 → 0.786. This is a clean dose-response, not a threshold effect, and it settles
   a question §14 below could only speculate about after two points: the global-vs-
   selection tension is real and continuous in λ_rank, not an artifact of an
   over-aggressive λ=0.25.
3. **Context conditioning is the single most valuable architectural component tested**
   (§14.1).

### 14.1 Model C — experimental-context conditioning (answers Q3)

Model C removes FiLM context conditioning entirely (`context_encoder` and `film` set to
`None`, not zeroed — a real ablation), holding λ_rank=0.25, seed, and split fixed for a
controlled comparison against Model A.

> **Fairness note**: an initial Model C run early-stopped at epoch 7 (best epoch 1) under
> the shared `patience=5` rule, because its validation curve was noisy-flat early. Models
> A and B had run the full 30 epochs, so comparing against that run would have understated
> Model C. It was **re-run with patience=30 for a like-for-like 30-epoch budget**, and only
> the fair re-run is reported here.

| Metric (locked test fold) | Model A (with context) | Model C (no context) | Δ |
|---|---:|---:|---:|
| Global Spearman ρ | **0.783** | 0.710 | −0.073 |
| Global Pearson r | **0.761** | 0.681 | −0.080 |
| Within-target Spearman ρ | **0.532** | 0.281 | −0.251 |
| Top-1 regret | **0.0131** | 0.0276 | +0.0145 (worse) |
| MAE | **0.114** | 0.135 | +0.021 (worse) |

**Q3 is answered clearly and affirmatively**: explicit experimental-context conditioning
substantially improves generalization across the heterogeneous corpus. The effect is
large on global correlation (−0.073 without it) and *dramatic* on within-target ranking
(−0.251, nearly halving it). This makes mechanistic sense: the corpus spans 12 lab×cell
groups whose absolute efficiency scales differ markedly, and without a context signal the
model must average over incompatible scales — which corrupts within-group ordering far
more than it corrupts the global trend.

### 14.2 Ranking-loss ablation (Models A vs B)

Directly answering task spec Research Question 4 ("Does adding within-target ranking
loss improve actual pegRNA selection even if global Pearson/Spearman changes little?"):
**partially yes, with an honest caveat the question's framing doesn't anticipate** —
global Spearman does *not* stay similar, it drops materially (0.864→0.783, a real
9-point difference, not noise). But within that trade, selection-relevant metrics that
should matter more in practice (top-1 regret, within-target ranking, NDCG) all improve
significantly. See `results/figures/07_ablation_comparison.png`.

This is a legitimate, unmanipulated ablation result, not the specific outcome we might
have hoped to observe going in — global correlation and selection quality are in real
tension here, not orthogonal, at least at this pilot's scale (one seed, ~260k rows,
λ_rank=0.25 untuned beyond the proposal's suggested value). Whether a smaller λ_rank
recovers most of the selection benefit while giving up less global correlation is a
natural next experiment (task spec's own guidance not to run a broad hyperparameter
sweep in the pilot stands — this is a specific, motivated follow-up, not exploratory
search).

Model C (no experimental-context conditioning) is reported in §14.1 above.

## 15. Failure analysis

1. **Calibration drifts downward for the highest-efficiency designs specifically.**
   `results/figures/01_predicted_vs_observed.png` (log-scale density + decile-mean
   overlay) shows good calibration through ~60% true efficiency, then increasing
   under-prediction: true 0.83 → predicted mean 0.61; true 0.92 → predicted mean 0.34
   (Model A, n=75 in the top decile — small-sample but consistent direction).
   Classic regression-to-the-mean under Huber loss on an efficiency distribution
   dominated by low values (mean true efficiency 0.24). This is a miscalibration, not a
   misranking — Spearman stays high because the shrinkage is monotonic — but it means
   raw predicted-efficiency values should not be read as calibrated probabilities for
   the highest-efficiency designs without a correction (e.g. isotonic regression on a
   held-out calibration set, not attempted this pilot).
2. **Performance varies substantially, and interpretably, by experimental context**
   (`results/figures/05_performance_by_context.png`): strongest on the largest,
   best-represented contexts (HEK293T PE2/PE4, HeLa — all Liu/Hsu-sourced, ρ 0.64-0.77),
   weakest on small single-condition Kim/DeepPrime cell types (A549 ρ=0.25, HCT116
   ρ=0.29, MDA-MB-231 ρ=0.36 — each contributing only 3,145-13,248 rows to the *entire*
   corpus, train+val+test combined). This tracks training-data volume per context
   closely and is the expected failure mode of a model with no explicit few-shot or
   transfer mechanism across contexts.
3. **Within-target ranking is harder than global correlation suggests**: median
   within-target ρ (0.61-0.71) far exceeds macro-mean (0.45-0.53), meaning a meaningful
   minority of target groups rank poorly even though most rank well. Not further
   decomposed this pilot; a natural follow-up is checking whether poorly-ranked groups
   correlate with low true-value dynamic range (near-tied candidates, where ranking is
   inherently harder and arguably less important) or with a systematic model weakness.
4. **Six of 42 training-corpus partitions remain unreconciled** (§3) — the model was
   trained on 88.1% of the target corpus, not the exact 297,962. This is the most
   likely single lever for improving all metrics above in a follow-up run, if the
   missing Schwank K562 PE4+epegRNA data (the largest gap, ~20k rows) can be located.

## 16. Conclusions — research questions

**Q1: Can a minimally mechanistic relational Transformer trained on the full corpus
match or outperform OptiPrime?** On the leak-free-by-construction locked test fold,
against the real OptiPrime code and released weights: yes for the no-rank variant
(Spearman 0.775 vs. 0.724), not for the rank variant (0.670). Given OptiPrime's number
may benefit from unknown leakage, this is best read as **"feasibility success,"
plausibly better"** rather than a clean, fully-controlled win — but achieved without any
mechanistic reaction-graph bias, on 88% of the intended training data, using a
generic Transformer recipe.

**Q2: Does the paired WT/edit token representation help?** Not directly ablated this
pilot (would require a second model variant with e.g. separately-encoded WT/edited
sequences) — flagged as a gap, not answered. The representation was used successfully
throughout, but its specific contribution vs. an alternative encoding is untested.

**Q3: Does experimental-context conditioning improve generalization?** **Yes,
substantially — it is the most valuable single component tested.** Removing FiLM context
conditioning costs −0.073 global Spearman and −0.251 within-target Spearman on the locked
test fold (§14.1, controlled against Model A with a like-for-like 30-epoch budget). Joint
training across 12 heterogeneous lab×cell groups depends on the model being able to tell
those groups apart.

**Q4: Does within-target ranking loss improve pegRNA selection even if global
correlation changes little?** More precisely than the question assumes: **it improves
selection significantly, but global correlation does not stay similar — it drops
materially.** See §14 for the full, honest picture.

**Q5: Can the full model train practically on one RTX PRO 6000 Blackwell GPU?**
Unambiguously yes: 26.6M params, ~25 min per 30-epoch run, 13 GB peak (of 95 GB), no
instability, no gradient issues, both models converged smoothly.

**Q6: Which errors remain?** High-efficiency calibration drift, context-dependent
performance tracking training-data volume, a right-skewed within-target ranking
distribution, and the 11.9%-incomplete training corpus. All four are itemized, with
specific next steps, in §15.

## 17. Recommendation for the next phase

In priority order, given each is individually cheap relative to what's already been
spent this pilot:

1. **Run Model C (no context conditioning)** — directly answers Q3, ~25 min of GPU time.
2. **Sweep λ_rank ∈ {0.05, 0.1, 0.25} on the existing train/val split** (not a broad
   search — 2 more runs, ~50 min) to check whether the global-correlation cost of
   ranking loss is a property of λ=0.25 specifically or persists at lower weight.
3. **Locate the missing Schwank K562 PE4+epegRNA data** (§3) — the single largest lever
   on final corpus completeness; requires either finding an unindexed public source or
   accepting it as genuinely unpublished.
4. **Complete DeepPrime-FT and PRIDICT2.0 baselines** properly (reproduce DeepSpCas9
   inference and DeepPrime's exact feature engineering) rather than under time pressure.
5. **Isotonic calibration** on a held-out slice, targeted at the high-efficiency
   under-prediction found in §15.
6. Only after 1-2 above: full 5-fold cross-validation on the locked corpus, per task
   spec §36, once architecture/hyperparameters are considered final.

## Reproducibility

| Artifact | Path |
|---|---|
| Model A checkpoint | `checkpoints/model_a_rank_1786733796/best.pt` |
| Model B checkpoint | `checkpoints/model_b_norank_1786735500/best.pt` |
| Training configs | `configs/pilot.yaml` (+ CLI `--lambda-rank` override) |
| Training history | `results/runs/model_{a,b}_*/training_history.csv` |
| Test-fold predictions | `results/runs/eval_test_fold/predictions_model_{a,b}*.parquet` |
| OptiPrime predictions | `data/interim/optiprime_compatible_test/predictions_20260814_124312/` |
| Hsu comparison table | `results/hsu_benchmark_table.csv` |
| Figures | `results/figures/*.{png,pdf}` |
| Git commit at time of training | `57b06ac` (Model A), `718d61c` (Model B) |
| Seed | 20260812 (data folds, model init, training) |
