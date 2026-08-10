#!/usr/bin/env bash
# Phase-0 smoke: convert 1–few clouds and run Waymo LitePT-S segmentation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv_smoke/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing $PY — create venv_smoke first (system torch + deps)." >&2
  exit 1
fi

"$PY" tools/convert_few_clouds_for_smoke.py --no-hf-demo \
  --from-bin /data/rawdata/lidar/00/00/ca7571770d71f9f231148f36feb5.bin \
  --from-bin /data/rawdata/lidar/00/00/25e15fe13a4fc98fbb33cce47a4d.bin \
  --max-points 120000

"$PY" tools/smoke_seg_infer.py \
  --data-root data/waymo_smoke \
  --device cuda \
  --no-amp \
  --save-pred

echo "Phase-0 smoke finished OK."
