#!/usr/bin/env bash
# Round-3: train Family A (layerwise context conditioning) on the remaining official
# folds so it can join the ensemble with proper out-of-fold evaluation.
#
# Rationale (see reports/round3_research_log.md): the ensemble gain comes from
# ARCHITECTURAL decorrelation, not from training-strategy variation. Measured
# rank-prediction correlations on dev fold 1:
#   round-1 vs domain-adapted : 0.9970  -> adds nothing, redundant
#   round-1 vs Family C       : 0.9550  -> adds +0.009 when blended
# Family A conditions on context at every Transformer block (rather than once after
# pooling) and is trained from scratch, so it should decorrelate like Family C does.
# Fold 1 already exists from round-2 Stage A (r2_familyA_layerwise_1787077501),
# which is why this script covers only the folds passed in FOLDS.
#
# --val-sources restricts checkpoint selection to Liu+Kim, matching the benchmark
# (round-3 §7) rather than the Schwank-heavy official fold.
set -euo pipefail
cd /srv/disk01/xhx/git/PEFormer
GPU="${GPU:-6}"; FOLDS="${FOLDS:?set FOLDS}"
export CUDA_VISIBLE_DEVICES="$GPU"
for k in $FOLDS; do
  .venv/bin/python scripts/train/train_pilot.py \
    --config "configs/official_cv${k}.yaml" \
    --run-name "r3_familyA_cv${k}" \
    --simplex-head --lambda-rank 0 --context-strategy layerwise \
    --val-sources hsu2026 deepprime \
    --patience 30 --seed 20260812
done
echo "FAMILY A FOLDS ${FOLDS} COMPLETE"
