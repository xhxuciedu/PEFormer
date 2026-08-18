#!/usr/bin/env bash
# Retrain the 6-model ensemble on the bar9-expanded corpus (285,260 rows).
# Composition matches the ensemble selected on validation in the first pilot:
# 3 simplex-head seeds + 3 logit-space scalar-head seeds, all lambda_rank=0.
set -euo pipefail

cd /srv/disk01/xhx/git/PEFormer
export CUDA_VISIBLE_DEVICES=6
PY=.venv/bin/python

for seed in 20260812 20260813 20260814; do
  $PY scripts/train/train_pilot.py --run-name "v2_simplex_s${seed}" \
    --simplex-head --lambda-rank 0 --seed "$seed" --patience 30
done

for seed in 20260812 20260813 20260814; do
  $PY scripts/train/train_pilot.py --run-name "v2_logit_s${seed}" \
    --regression-space logit --lambda-rank 0 --seed "$seed" --patience 30
done

echo "ALL 6 ENSEMBLE MEMBERS COMPLETE"
