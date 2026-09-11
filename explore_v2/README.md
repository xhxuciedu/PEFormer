# explore_v2 — executing the research plan's section 8A

This directory is the record of running `explore_v2_research_plan.md`'s two-week
current-data programme, plus the computational parts of its two-month programme.
**Read `SUMMARY_REPORT.md` first** — one consolidated report over all fifteen experiments,
with the verdict and the recommended next steps. `RESULT.md` is the headline result in
detail. `GATE_A_REPORT.md` is the
go/no-go decision on research direction and cites everything else. `WORK_SUMMARY.md` records what was built, what was decided along the way, and what
was deliberately not done. **`CORRECTIONS.md` is the correction of record**: a later review
found real defects in E04, and two published numbers and one conclusion changed.

No model was trained. The frozen ordinal-S4D backbone is used out of fold: round 4's five
checkpoints each held one official fold out, so every row is embedded and scored by the
checkpoint that never saw its fold (`extract_embeddings.py`). Held-out fold 0 was used for
audit and descriptive re-scoring only.

## Scripts, in the order they were run

| script | what it answers |
|---|---|
| `canon.py` | canonical intended-allele keys from the corpus's variable-length windows: minimal edit, indel left-normalisation, strand normalisation, flanks read off the site's longest observed window |
| `extract_embeddings.py` | out-of-fold pre-FiLM pooled representations and predictions for all 318,471 rows (GPU, a few minutes) |
| `e01_decision_manifest.py` | how many "one allele, one context, choose a pegRNA" groups the corpus actually contains, at what depth, on which fold; replicate and fold-purity audit |
| `e02_corrected_metrics.py` | the frozen predictions re-scored on that decision, four groupings, plus the pairwise estimand fold 0 can support |
| `e03_matched_support.py` | the design-by-context interaction on matched candidate sets: variance components against a replicate-derived noise model, reversal rates against an additive null, and what a context-blind chooser loses |
| `e04_interaction_predictability.py` | whether the interaction is predictable at held-out loci, against additive, no-context and structured-shuffle controls, with an injected-interaction positive control |
| `e05_reference_panel.py` | the plan's priority 1: does measuring 12-192 reference designs in an unseen context improve selection there? |
| `e06_external_eligibility.py` | what external surface exists on disk, what must be acquired, and one reserved panel |
| `e07_context_correction.py` | does an explicit context-conditioned design correction improve selection for contexts that *are* in training? |
| `e08_pilot_power.py` | the plan's pilot power, from measured variability instead of an illustrative SD |
| `e09_clean_interaction_retest.py` | E04's claim retested with one coordinate system, no encoder exposure, and representation-matched controls — the repair demanded by the review |
| `e10_build_reserved_panel.py` | rebuilds Kim's large library into both models' input schemas, with a three-way validated sequence reconstruction, and removes every overlap with training |
| `precompute_rs3_panel.py` | pre-populates OptiPrime's RuleSet3Score disk cache for the panel (isolated rs3 environment) |
| `run_optiprime_panel.sh` + `collect_optiprime_panel.py` | runs the released OptiPrime code and all five released fold checkpoints on the panel, then joins the chunked predictions |
| `e11_score_reserved_panel.py` | **the pre-declared comparison on the panel**: achieved efficiency, best-design hit rate, regret, threshold success, screening effort, by depth and edit type |
| `e12_context_sensitivity.py` | how much the panel endpoints move if a different plausible experimental context had been assigned |
| `e13_regroup_training.py` | measures that the pairwise ranking loss is grouped on a design artefact and sees 3.9% of the available within-allele comparisons; writes a re-grouped corpus differing only in `group_key` |
| `e14_eval_regroup.py` | the two training arms on fold 0's canonical decision |
| `e15_regroup_on_panel.py` | the same two arms on the reserved panel, alongside OptiPrime — a disclosed second use of the panel |
| `verify_report.py` | 45 numbers asserted in `GATE_A_REPORT.md`, each checked against the JSON that produced it |

Each `eNN_*.py` writes `eNN_*.json` (with git commit, seed and input hashes) and
`eNN_*.md`. Re-running any of them regenerates both.

## Reproducing

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_v2/extract_embeddings.py
for s in e01_decision_manifest e02_corrected_metrics e03_matched_support \
         e04_interaction_predictability e05_reference_panel e06_external_eligibility \
         e07_context_correction e08_pilot_power e09_clean_interaction_retest \
         e10_build_reserved_panel; do
  env PYTHONPATH=src .venv/bin/python "explore_v2/$s.py"
done
.venv/bin/python explore_v2/verify_report.py
```

`e01` must run before the rest: it writes `cache/decision_manifest.parquet`, which
everything downstream keys on. `e03` caches the 110,321-row quartet table that `e04`
consumes; building it takes about eight minutes, reading it takes seconds. `e02` writes
`e02_pairwise_table.csv`, which `e08` reads. Wall time for the whole set, after the
embeddings: roughly an hour, most of it `e05`'s 364 episodes.

Untracked build products live in `cache/` (embeddings are ~470 MB) and are safe to delete;
every script rebuilds what it needs.

**The caches are versioned on the canonicaliser.** `canon.CANON_VERSION` is part of the
manifest filename, and every downstream script reads through `_v2common.require_manifest()`,
which raises rather than falling back if its version is missing. The published E01–E09
numbers were produced with `CANON_VERSION = 1`, whose indel keys were not strand-invariant
(`CORRECTIONS.md` C3); the current code is version 2, so re-running anything downstream
requires rebuilding the manifest with E01 first. Set `CANON_VERSION = 1` to reproduce the
published numbers exactly.

The panel comparison has its own sequence, because it depends on an isolated environment
and on the released OptiPrime code:

```bash
<rs3env>/bin/python explore_v2/precompute_rs3_panel.py     # RuleSet3 cache, ~10 s
bash explore_v2/run_optiprime_panel.sh                    # released OptiPrime, ~3 h CPU
.venv/bin/python explore_v2/collect_optiprime_panel.py
env PYTHONPATH=src .venv/bin/python explore_v2/e11_score_reserved_panel.py \
  --op-predictions explore_v2/cache/optiprime_panel_predictions.parquet
env PYTHONPATH=src .venv/bin/python explore_v2/e12_context_sensitivity.py
```

Give each parallel OptiPrime chunk its **own** `_disk_cache` copy. OptiPrime writes derived
features back into that cache, so concurrent writers to one shared directory corrupt the
pickle shards; two chunks failed that way before this was fixed.

## Do not open without a plan

`reserved_panel_v2.parquet` is 30,475 fixed-allele decision groups from Kim's large library,
which is not in OptiPrime's training mix. `PREREGISTRATION_reserved_panel.md` declared what
would be computed on it, and the file was hashed before any model saw it. **That declared
comparison has now been run once, in E11.** The panel is therefore spent for confirmatory
purposes: any further model choice informed by it makes it development data, and results
from a model tuned afterwards must say so. Reserve a different dataset for the next
confirmation.
