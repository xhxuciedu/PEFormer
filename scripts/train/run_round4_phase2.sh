#!/bin/bash
# Round-4 Phase 2: train each promoted member on the official 5-fold split.
#
# Phase-1 screening trained on a dev fold and picked the best epoch on that same
# fold's validation rows, so its standalone and S3 numbers are optimistically biased.
# The existing ensemble members are out-of-fold on the official split and carry no
# such bias, which makes the Phase-1 comparison useful for *ranking* candidates
# (they all share the bias) but not for deciding what a member is worth. Training
# each promoted member on all five official folds puts it on exactly the same
# footing as the incumbents, so Phase 3 can compare them honestly.
#
# 5 members x 5 folds = 25 runs, packed onto the three large cards (the 11GB cards
# cannot hold this recipe at batch 512, and switching them to batch 256 would make
# those folds a different recipe from their siblings).

set -u
cd "$(dirname "$0")/../.."
S="${SCRATCH_DIR:?set SCRATCH_DIR}"
CFG=configs/round3/baseline_round1.yaml
mkdir -p "$S/phase2"

# member:extra-flags
MEMBERS=(
  "ordSSM:--ordinal-head --sequence-mixer ssm"
  "ssm:--sequence-mixer ssm"
  "ordC:--feature-branch --ordinal-head"
  "ordA:--context-strategy layerwise --ordinal-head"
  "ordB:--ordinal-head"
)

# Slots, sized by memory: the recipe needs ~15GB, SSM runs somewhat more.
SLOTS=(6 6 6 6 6 7 7 7 2 2 2)

jobs=()
for m in "${MEMBERS[@]}"; do
  name="${m%%:*}"; flags="${m#*:}"
  for k in 1 2 3 4 5; do jobs+=("$name|$flags|$k"); done
done

nslots=${#SLOTS[@]}
for i in "${!SLOTS[@]}"; do
  gpu="${SLOTS[$i]}"
  # Each slot runs its share of the queue strictly sequentially, so a slot never
  # holds two models' worth of memory at once.
  (
    for ((j=i; j<${#jobs[@]}; j+=nslots)); do
      IFS='|' read -r name flags k <<< "${jobs[$j]}"
      run="r4p2_${name}_cv${k}"
      echo "[slot $i gpu $gpu] START $run $(date +%T)"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src .venv/bin/python scripts/train/train_pilot.py \
        --config "$CFG" --run-name "$run" --val-fold "$k" $flags \
        > "$S/phase2/$run.log" 2>&1
      echo "[slot $i gpu $gpu] DONE  $run rc=$? $(date +%T)"
    done
  ) &
done
wait
echo "PHASE2 COMPLETE"
