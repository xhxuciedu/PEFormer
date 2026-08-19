#!/usr/bin/env bash
# Round-3 Phase 1 (spec §8): domain-adaptive fine-tuning of the round-1 official
# checkpoints on Liu+Kim only.
#
# Cleanliness: round-1 checkpoint k was trained on official folds {1..5}\{k}. The
# fine-tune uses configs/official_cv{k}.yaml, i.e. the SAME train_folds, so fold k
# is unseen in both stages. OOF evaluation on dev-fold rows belonging to official
# fold k is therefore uncontaminated, and the 5 resulting checkpoints are directly
# usable as a final ensemble member.
#
# Validation is restricted to Liu+Kim (--val-sources) so best-checkpoint selection
# optimises the actual benchmark, not the Schwank-heavy official fold.
set -euo pipefail
cd /srv/disk01/xhx/git/PEFormer
LR="${LR:?set LR}"; TAG="${TAG:?set TAG}"; GPU="${GPU:-6}"; FOLDS="${FOLDS:?set FOLDS}"
export CUDA_VISIBLE_DEVICES="$GPU"
for k in $FOLDS; do
  CKPT=$(ls -d checkpoints/cv${k}_simplex_*/best.pt)
  .venv/bin/python scripts/train/train_pilot.py \
    --config "configs/official_cv${k}.yaml" \
    --run-name "r3_dapt_${TAG}_cv${k}" \
    --init-from "$CKPT" \
    --simplex-head --lambda-rank 0 \
    --train-sources hsu2026 deepprime \
    --val-sources hsu2026 deepprime \
    --lr "$LR" --max-epochs 10 --patience 10 --seed 20260812
done
echo "DAPT ${TAG} COMPLETE (folds ${FOLDS})"
