# V3.1 computational adaptation research

This directory implements the bounded first tranche of
[the next-step plan](../explore_v3/NEXT_STEPS.md). Historical v2/v3 artifacts are
read-only dependencies. See the [frozen protocol](EXECUTION_PROTOCOL.md),
[execution/correction notes](RUNTIME_NOTES.md), and
[independent-data feasibility](DATA_FEASIBILITY.md).

**The 76-fit bounded study is complete:** see [the research report](RESEARCH_REPORT.md).
The 32-fit screen is archived with [its promotion gate](SCREEN_REPORT.md).
The locked 48-configuration acquisition replication completed with 44 new fits
and four exact screen reuses: [replication results](REPLICATION_RESULTS.md).
The larger-budget/architecture branches were deferred at the final gate, not
executed or counted as negative results. `M` in these files means
balanced teacher-margin preservation, not the earlier v2 multi-task arm.

## Reproduction

Use the repository's existing Python environment, from the repository root.
Commands fail rather than overwrite completed output. GPUs shown below were
available at launch time; check availability again before reproduction.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/diagnose.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest explore_v3_1/test_followup.py explore_v3/test_protocol.py explore_v2/test_adaptation_followup.py explore_v2/test_endpoints.py -q
.venv/bin/python -u explore_v3_1/run_stage.py --stage controls --gpu 7 --frozen-gpu 0
.venv/bin/python explore_v3_1/select_e_recipe.py
.venv/bin/python explore_v3_1/analyze.py --stage controls
.venv/bin/python -u explore_v3_1/run_stage.py --stage factorial --gpu 0
.venv/bin/python explore_v3_1/analyze.py --stage factorial
.venv/bin/python explore_v3_1/verify.py --expected 32
```

For the original execution, the factorial began after the four E controls and
their inner-only recipe selection completed, while the remaining conventional
controls finished on the other GPU. The sequential commands above produce the
same experiment matrix. `e_recipe.json` and full `control_selection.json` must
agree. No factorial recipe is selected from outer validation or source fold 0.

The initial `analyze.py` and `verify.py` commands deliberately expect exactly
the first-stage matrix. Run them before replication adds further fits; their
completed screen outputs are preserved. Do not rerun them on the expanded
run directory and call the extra acquisitions new hyperparameter candidates.

After the recorded positive `REPLICATION_DECISION.json`:

```bash
.venv/bin/python -u explore_v3_1/run_replication.py --gpus 0 1 3 7
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/analyze_replication.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/verify_replication.py
.venv/bin/python explore_v3_1/make_manuscript_table.py
.venv/bin/python explore_v3_1/make_manuscript_table.py --check
```

The [replication protocol](REPLICATION_PROTOCOL.md) fixes all recipes, subsets,
optimizer seeds and checkpoint rules before launch. No larger-budget or
architecture search is part of this tranche. The manuscript-table generator
requires completed independent endpoint/selection verification.

Additional read-only analyses of completed models:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/secondary_analysis.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/selection_stability.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/error_strata.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/acquisition_audit.py
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python explore_v3_1/benchmark_inference.py --physical-gpu 4
```

The latency benchmark uses full encoder/readout inference on identical
pretokenized GPU-resident inputs, not cached-feature-only inference. It is not
a matched speed comparison against OptiPrime. All diagnostic output is
exploratory; measured outcome gaps are not replicate-based uncertainty.

Manuscript checks, in addition to the repository's original numerical checker:

```bash
.venv/bin/python explore_v3_1/verify_followup_prose.py
.venv/bin/python explore_v3_1/make_manuscript_table.py --check
```

The updated manuscript compiles to
`explore_v3_1/cache/manuscript/pe_rankformer_paper.pdf`; the original PDF was
not overwritten. A tracked publication snapshot is available as
[pe_rankformer_paper_v31.pdf](../reports/paper/pe_rankformer_paper_v31.pdf).
Protocols here are locally prespecified documents, not a
public-registry preregistration.

Additional data checks (network access for the first command only):

```bash
.venv/bin/python -u explore_v3_1/acquire_metadata.py
.venv/bin/python explore_v3_1/audit_external.py
.venv/bin/python explore_v3_1/audit_alphabet.py
.venv/bin/python explore_v3_1/audit_core_overlap.py
```

`alphabet_audit.json` explicitly corrects the first external core-homology pass;
use its populations and `data/epridict_input_audit_alphabet_corrected.parquet`,
not the superseded DNA-only core count. No external panel is scored by these
commands. Public acquisition downloads metadata and one small workbook only,
not sequencing reads. See `DATA_PROTOCOL.md` for outcome-independent rules.

## Interpretation and artifacts

Each run saves target-only, source-constrained and final-horizon checkpoints;
all inner/source-validation checkpoint scores; training traces; per-candidate
evaluation scores; and full fingerprints. Source-constrained selection is a
point-estimate per-study filter, not a statistical noninferiority guarantee.
Reports use the same deployed score on source and target.

The label budget includes inner validation, but not all labels used in historical
method development. Screen results are one acquisition subset and one optimizer
seed. Kim outer validation and source fold 0 remain exposed development/audit
surfaces. No target-test or independent-confirmation score is produced.

GPU/network access may require environment approval. Do not stop unrelated jobs.
Model artifacts and sequence-bearing files remain local and ignored by git;
do not redistribute the privately supplied source corpus or its sequence-level
derivatives without permission. New temporary files use the workspace cache.
