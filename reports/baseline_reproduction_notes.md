# Baseline reproduction notes

## OptiPrime — real inference achieved

Ran the actual released OptiPrime code and weights (`external/optiprime`, all 5
released fold checkpoints ensembled, matching `PREDICT_PE.py`'s own averaging) on the
locked Hsu test-fold rows (15,022 rows, fold 0).

### Environment issues resolved

1. **`rs3` (RuleSet3 Cas9 on-target scoring)** requires `scikit-learn<=1.0.2`, which
   fails to build from source under Python 3.11 with our numpy/torch stack. Resolved by
   creating an isolated `conda` env (Python 3.10, where scikit-learn 1.0.2 has a
   prebuilt wheel), computing `RuleSet3Score` for the 233 unique test-fold spacers
   there, and pre-populating OptiPrime's own on-disk hash cache
   (`_disk_cache/RuleSet3Score/{hash2}_{DATA,META}.pkl`) so the real inference code
   never needs to call `rs3` at runtime. The `rs3` import itself was satisfied with a
   stub package tracked at `scripts/evaluate/rs3_stub/rs3/` (put ahead of
   `external/optiprime` on `PYTHONPATH`) that raises if actually called -- a canary
   that the cache is incomplete, not a silent fallback.
   See `scripts/evaluate/precompute_ruleset3_cache.py`. Reproduce with:
   `PYTHONPATH=external/optiprime:scripts/evaluate/rs3_stub .venv/bin/python external/optiprime/PREDICT_PE.py ...`
2. **`tensorflow`** (used only for `tf.data` batching in `make_loader`, not for any
   model computation) was missing; installed `tensorflow-cpu` into `.venv`.
3. No CUDA-enabled `jaxlib` is installed, so OptiPrime inference runs on CPU. This is
   acceptable for a one-off ~15k-row evaluation (~minutes) but was not worth setting up
   a second JAX/CUDA stack alongside the PyTorch one for a single baseline run.

### The 4bp-upstream-context problem

OptiPrime's `format_pe_df` requires `full_unedited` to place the 20nt protospacer at a
fixed offset (`PS20_OFFSET=4`), with `proto30 = full_unedited[:30]` exactly 30nt — see
`reports/optiprime_data_loader_reverse_engineering.md` §6. The public Hsu Supplementary
workbook does not provide this upstream context for Lib-MMR designs (protospacer starts
at offset 0) and provides only partial context for some Lib-CV designs (offset 3, one
short of the required 4).

**Resolution**: left-pad with a fixed, explicitly non-genomic filler (`'A' * k`,
`k` = 1 or 4) to reach the required offset (`scripts/evaluate/build_optiprime_test_csv.py`).
An earlier attempt used `'N'` as the filler, which is more honest about "we don't know
this base" but **breaks OptiPrime's own sequence encoder**
(`scripts/pe/pe_utils.py::seq_encoding` one-hots strictly over `{A,C,G,T}` — confirmed
by the actual crash: `ValueError: 'N' is not in list`). Switched to a fixed `'A'`
filler, which is arbitrary but at least lets the real code path run.

**Consequence**: for the 7,319 rows needing full 4bp padding (49% of the test set,
concentrated in Lib-MMR) and 1,279 needing 1bp padding, OptiPrime's PAM-adjacent and
upstream-homology-arm features are computed over placeholder sequence, not real
genomic context. Reported OptiPrime metrics on this test set should be read as an
approximation with a known, quantified source of extra noise (roughly half the test
set), not a faithful reproduction of what OptiPrime would predict with true genomic
context. This is flagged in every table that includes these numbers.

### Leakage caveat (task spec §29, §45)

We do not have OptiPrime's original protospacer-stratified fold assignments (the
`split` column is baked into unpublished training CSVs — see the loader
reverse-engineering report §5). The publicly released 5 fold checkpoints were therefore
almost certainly trained on folds that are **not identical** to our own fold 0. Some
fraction of our "held-out" test rows may have been in the training set for one or more
of the 5 released OptiPrime models. This means the OptiPrime numbers reported here are
better characterized as **in-sample-contaminated retrospective predictions**, not a
strictly leak-free held-out evaluation of OptiPrime. We ensemble all 5 released models
(as `PREDICT_PE.py` does) to reduce (not eliminate) the chance that any single row was
seen by every model. PE-RankFormer's own numbers on this same test fold are leak-free
by construction (fold 0 was never touched during training or model selection).

## DeepPrime-FT and PRIDICT2.0 — attempted, deferred with specific reasons

Both were investigated (not just skipped outright) before deciding not to complete
inference in this session.

**DeepPrime-FT**: `external/deepprime/models/ontarget_variants/` has exactly the
per-condition fine-tuned checkpoints the paper's Fig. 4 caption specifies (HEK293T
PE2max-e / PE4max-e, HeLa PE2max — reused for HeLa PE4max since no HeLa PE4max-e
checkpoint exists, matching the paper's own stated fallback). The blocker is not
loading the checkpoints (`utils/model.py::GeneInteractionModel`, standard PyTorch) but
reproducing DeepPrime's **input features exactly**: `utils/data.py::select_cols` needs
24 hand-engineered columns (four melting temperatures, GC content/counts, MFE3/4 via
ViennaRNA, and critically `DeepSpCas9_score`, a *separate* pretrained on-target cutting
model). Getting the Tm/MFE formulas byte-for-byte consistent with DeepPrime's own
preprocessing, and standing up DeepSpCas9 inference correctly, is real additional work
with meaningful risk of a subtle mismatch producing misleading numbers — worse than not
reporting DeepPrime-FT at all.

**PRIDICT2.0**: same shape of problem — `external/pridict2/trained_models/DeepCas9_TestCode.py`
shows it depends on the same DeepSpCas9 on-target scoring model as an input feature.

**Decision**: given the task spec's explicit priority ordering (§51: "data correctness,
fair evaluation, one strong model, clean implementation, reproducibility... over broad
exploratory model search") and that a wrong/mismatched reimplementation is worse than an
honestly-labeled gap, these two baselines are left as documented follow-up work rather
than attempted under time pressure. The comparison that **is** complete — PE-RankFormer
vs. the real OptiPrime code and weights on the locked Hsu test fold — is the specific
comparison the paper itself treats as primary (Fig. 4a), so the pilot's central question
is still answerable without them.
