#!/usr/bin/env bash
# Stage B (round-2 spec section 6): full 5-fold CV confirmation for Family C
# (feature branch), the Stage-A winner. Fold 1 already exists as
# r2_familyC_features (val_fold=1); this runs the remaining folds 2-5.
set -euo pipefail
cd /srv/disk01/xhx/git/PEFormer
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
PY=.venv/bin/python
for k in 2 3 4 5; do
  $PY scripts/train/train_pilot.py --config "configs/official_cv${k}.yaml" \
    --run-name "r2_familyC_cv${k}" --simplex-head --lambda-rank 0 --feature-branch --seed 20260812
done
echo "FAMILY C STAGE-B CV COMPLETE (folds 2-5)"
