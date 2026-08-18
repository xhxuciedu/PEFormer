#!/usr/bin/env bash
# Match OptiPrime's own protocol exactly: 5 cross-validation models, model k holding out
# split k for validation/early-stopping and training on the other four. Ensembling all 5
# means the ensemble collectively sees all 297,962 training rows, exactly as OptiPrime's
# 5 released checkpoints do. The previous 3-model run used only splits 2-5 (80% of the
# data) and so was handicapped relative to the baseline.
set -euo pipefail
cd /srv/disk01/xhx/git/PEFormer
export CUDA_VISIBLE_DEVICES=6
PY=.venv/bin/python
for k in 1 2 3 4 5; do
  train=$(python3 -c "print([f for f in [1,2,3,4,5] if f != $k])")
  sed -e "s|val_fold: 1|val_fold: $k|" -e "s|train_folds: \[2, 3, 4, 5\]|train_folds: $train|" \
      configs/official.yaml > "configs/official_cv$k.yaml"
  $PY scripts/train/train_pilot.py --config "configs/official_cv$k.yaml" \
    --run-name "cv${k}_simplex" --simplex-head --lambda-rank 0 --seed 20260812 --patience 30
done
echo "CV5 ENSEMBLE COMPLETE"
