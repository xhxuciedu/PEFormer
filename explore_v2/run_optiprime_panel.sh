#!/bin/bash
# Run the released OptiPrime code and weights on the reserved panel.
#
# Follows reports/baseline_reproduction_notes.md exactly: all five released fold
# checkpoints (PREDICT_PE.py averages them itself), the rs3 import satisfied by the
# canary stub so a cache miss raises instead of silently falling back, and the
# RuleSet3Score disk cache pre-populated by explore_v2/precompute_rs3_panel.py.
#
# Chunked because PREDICT_PE.py allocates a (5, n, 55, 55) float32 array of reaction
# matrices -- 7 GB at the full panel size -- which is written to disk and then discarded.
# The panel is one group (Kim_HEK293T) and one file, so chunking cannot change any
# prediction.
set -u
cd /srv/disk01/xhx/git/PEFormer
SRC=data/interim/reserved_panel_kim_large
WORK=${WORK:-/srv/disk01/xhx/tmp/claude-8385/-srv-disk01-xhx-git-PEFormer/c07d2d81-0766-40c2-8caf-61371c8f16e6/scratchpad/op_panel}
CHUNK=${CHUNK:-20000}
mkdir -p "$WORK"

.venv/bin/python - "$SRC" "$WORK" "$CHUNK" <<'PY'
import sys, pandas as pd, os
from pathlib import Path
src, work, chunk = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
d = pd.read_csv(src / "Kim_HEK293T_LibSmall_PE2max_test.csv")
for i in range(0, len(d), chunk):
    sub = d.iloc[i:i+chunk]
    dd = work / f"chunk{i//chunk:02d}"
    dd.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dd / "Kim_HEK293T_LibSmall_PE2max_test.csv", index=False)
    link = dd / "_disk_cache"
    if not link.exists():
        os.symlink((src / "_disk_cache").resolve(), link)
print("chunks:", len(list(work.glob("chunk*"))))
PY

for dd in "$WORK"/chunk*; do
  if [ -f "$dd/DONE" ]; then echo "skip $dd"; continue; fi
  echo "=== $dd $(date +%H:%M:%S)"
  PYTHONPATH=external/optiprime:scripts/evaluate/rs3_stub \
  JAX_PLATFORMS=cpu .venv/bin/python external/optiprime/PREDICT_PE.py \
    --data_dir "$dd" \
    --rx_graph external/optiprime/graphs/pe_model.rx \
    --weight_dirs external/optiprime/weights/model_1 external/optiprime/weights/model_2 \
                  external/optiprime/weights/model_3 external/optiprime/weights/model_4 \
                  external/optiprime/weights/model_5 \
    --batch_size 512 > "$dd/run.log" 2>&1
  rc=$?
  out=$(ls -d "$dd"/predictions_* 2>/dev/null | tail -1)
  if [ "$rc" -eq 0 ] && [ -n "$out" ] && [ -f "$out/predictions.csv" ]; then
    rm -f "$out/gmats.npy"; touch "$dd/DONE"; echo "  ok -> $out"
  else
    echo "  FAILED rc=$rc"; tail -25 "$dd/run.log"; exit 1
  fi
done
echo "OPTIPRIME_PANEL_COMPLETE $(date +%H:%M:%S)"
