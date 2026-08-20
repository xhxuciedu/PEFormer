#!/bin/bash
# Memory-aware job queue for round-4 training runs.
#
# Replaces the fixed-slot packing in run_round4_phase2.sh, which assumed every run
# needed ~15GB and OOM'd 9 of 25 jobs because the SSM runs actually need ~22GB.
# Rather than guessing a better slot count, this dispatches a job only when a GPU
# genuinely has room for it right now, and re-checks between dispatches.
#
# Also idempotent: a job whose checkpoint already exists is skipped, so it can be
# re-run after partial failure without retraining what already succeeded.
#
# Usage: JOBS_FILE=jobs.txt SCRATCH_DIR=... bash queue_runs.sh
#   each line of JOBS_FILE:  <run-name>|<extra flags>
# Env: NEED_MB (default 24000) memory to reserve per job; POLL (default 60).

set -u
cd "$(dirname "$0")/../.."
S="${SCRATCH_DIR:?set SCRATCH_DIR}"
JOBS_FILE="${JOBS_FILE:?set JOBS_FILE}"
CFG="${CFG:-configs/round3/baseline_round1.yaml}"
NEED_MB="${NEED_MB:-24000}"
POLL="${POLL:-60}"
GPUS="${GPUS:-2 6 7}"
mkdir -p "$S/phase2"

free_gpu() {
  # Emit the first GPU whose *free* memory exceeds NEED_MB. Reading actual free
  # memory rather than counting jobs is what makes this robust to runs of
  # different sizes sharing a card.
  for g in $GPUS; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits)
    if [ $((total - used)) -gt "$NEED_MB" ]; then echo "$g"; return; fi
  done
}

while IFS='|' read -r run flags; do
  [ -z "${run// }" ] && continue
  if compgen -G "checkpoints/${run}_*/best.pt" > /dev/null; then
    echo "SKIP $run (checkpoint exists)"; continue
  fi
  while :; do
    g=$(free_gpu)
    if [ -n "$g" ]; then break; fi
    sleep "$POLL"
  done
  echo "START $run on gpu $g $(date +%T)"
  CUDA_VISIBLE_DEVICES="$g" PYTHONPATH=src nohup .venv/bin/python scripts/train/train_pilot.py \
    --config "$CFG" --run-name "$run" $flags > "$S/phase2/$run.log" 2>&1 &
  # Let the new process actually claim its memory before the next capacity check,
  # otherwise the next iteration sees stale free memory and double-books the card.
  sleep 120
done < "$JOBS_FILE"

wait
echo "QUEUE COMPLETE"
