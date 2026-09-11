#!/usr/bin/env bash
# Regenerate the historical benchmark analyses and corrected utility table.
# Adaptation follow-ups have separate versioned protocols in explore_v3{,_1}.
#
# This script does NOT train. It re-derives all reported results from the frozen
# prediction files and checkpoints already in the repository, then rebuilds the tables
# and figures and verifies the manuscript against the artifacts. Training is a separate,
# much longer path documented in reproduce/README.md.
#
# Usage:  bash reproduce/run_all.sh
# Runtime: roughly 25 minutes on one CPU, no GPU required except where noted.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
step () { printf '\n=== %s\n' "$1"; }

step "0. environment and integrity"
$PY - <<'EOF'
import hashlib, json, subprocess, sys, torch
from pathlib import Path
paths = ["data/processed/optiprime_official_318471.parquet",
         "data/processed/featurized_official.npz",
         "data/processed/round3_dev_assignments.parquet",
         "results/round4/heldout/predictions_round4_final.parquet",
         "results/round5/heldout_calibrated.parquet",
         "results/heldout_full_head_to_head.parquet"]
env = {"python": sys.version.split()[0], "torch": torch.__version__,
       "cuda_available": torch.cuda.is_available(),
       "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()}
missing = [p for p in paths if not Path(p).exists()]
if missing:
    sys.exit("MISSING required artifacts:\n  " + "\n  ".join(missing))
h = {}
for p in paths:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 20):
            d.update(c)
    h[p] = d.hexdigest()[:16]
Path("reproduce/environment.json").write_text(json.dumps({"env": env, "sha256": h}, indent=2))
print("environment and input hashes -> reproduce/environment.json")
for k, v in env.items():
    print(f"  {k}: {v}")
EOF

step "1. unit tests (232 expected)"
$PY -m pytest tests/ -q

step "2. Phase 1 re-analyses on frozen predictions"
for t in revision/task_1_1_per_target.py \
         revision/task_1_2_utility.py \
         revision/task_1_3_tie_robust.py \
         revision/task_1_4_leakage_free.py \
         revision/task_1_5_calibration_baselines.py \
         revision/task_1_6_ceiling_uncertainty.py \
         revision/task_1_7_multiplicity.py \
         revision/task_1_8_weighting_by_partition.py \
         revision/task_2_round9_wave_results.py; do
  echo "--- $t"; $PY "$t" > /dev/null
done

step "3. supporting analyses"
$PY scripts/evaluate/stratified_comparison.py > /dev/null
$PY scripts/evaluate/gbm_baseline.py > /dev/null
$PY scripts/evaluate/early_stopping_survey.py > /dev/null
$PY revision/audit_00_consistency.py > /dev/null

step "4. manuscript tables, generated from the artifacts"
$PY revision/make_paper_tables.py
# The legacy generator above includes the retired protospacer utility estimand.
# Restore the corrected fixed-allele/context table using the existing v2 artifacts.
$PY explore_v2/make_utility_table.py

step "5. figures"
$PY scripts/evaluate/make_paper_figures.py

step "6. verify the manuscript prose against the artifacts"
$PY reproduce/verify_manuscript.py

step "7. compile the manuscript"
cd reports/paper
pdflatex -interaction=nonstopmode pe_rankformer_paper.tex > /dev/null
pdflatex -interaction=nonstopmode pe_rankformer_paper.tex > /tmp/pe_final.log
printf 'pages: %s  overfull: %s  undefined refs: %s\n' \
  "$(grep -o '([0-9]* pages' /tmp/pe_final.log | tail -1 | tr -d '(')" \
  "$(grep -c 'Overfull .hbox' /tmp/pe_final.log || true)" \
  "$(grep -c 'Reference.*undefined' /tmp/pe_final.log || true)"

printf '\nAll steps completed.\n'
