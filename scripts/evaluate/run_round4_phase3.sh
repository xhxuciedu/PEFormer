#!/bin/bash
# Round-4 Phase 3, step 1: unbiased OOF dev predictions for the promoted members.
#
# Each member has five checkpoints, one per official fold. --oof scores every row
# with only the checkpoint that held that row's official fold out, which is what
# makes these numbers comparable to the incumbents' -- and what Phase 1 could not
# do, since its candidates were best-epoch-selected on the very rows they scored.
#
# evaluate_on_devfolds.py hard-fails unless the five checkpoints cover folds 1-5
# exactly, so a member with a missing or duplicated fold cannot silently produce a
# partially in-sample prediction set.
#
# Writes predictions_r4p2_<member>_oof_<fold>.parquet for all three dev folds.

set -eu
cd "$(dirname "$0")/../.."
S="${SCRATCH_DIR:?set SCRATCH_DIR}"
GPU="${GPU:-7}"
mkdir -p "$S/phase3"

members_plain="ordSSM ssm ordA ordB"
members_feat="ordC"   # feature-branch models need --feature-branch at eval time too

for m in $members_plain $members_feat; do
  cks=$(ls -d checkpoints/r4p2_${m}_cv*/best.pt 2>/dev/null | sort)
  n=$(echo "$cks" | grep -c . || true)
  if [ "$n" -ne 5 ]; then
    echo "SKIP $m: found $n/5 checkpoints"; continue
  fi
  extra=""
  case " $members_feat " in *" $m "*) extra="--feature-branch";; esac
  echo "=== $m (5 checkpoints, OOF) ==="
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src .venv/bin/python \
    scripts/evaluate/evaluate_on_devfolds.py \
    --checkpoints $cks --model-name "r4p2_${m}_oof" --oof $extra \
    2>&1 | tee "$S/phase3/oof_${m}.log" | grep -E "OOF mode|mean|combined" || true
done
echo "PHASE3 OOF PREDICTIONS COMPLETE"
