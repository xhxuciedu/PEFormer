#!/usr/bin/env bash
# Train the ensemble on OptiPrime's OFFICIAL training mix (297,962 rows, authors-supplied),
# using their own split structure: val = split 1, train = splits 2-5, test = held-out.
set -euo pipefail
cd /srv/disk01/xhx/git/PEFormer
export CUDA_VISIBLE_DEVICES=6
PY=.venv/bin/python
for seed in 20260812 20260813 20260814; do
  $PY scripts/train/train_pilot.py --config configs/official.yaml \
    --run-name "off_simplex_s${seed}" --simplex-head --lambda-rank 0 --seed "$seed" --patience 30
done
echo "OFFICIAL ENSEMBLE COMPLETE"
