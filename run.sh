cd /home/dev/01develop/LitePT
export PYTHONPATH=./
export PATH="/usr/local/cuda/bin:${PATH}"

PY="${PWD}/.venv_smoke/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing $PY" >&2
  exit 1
fi

# "$PY" tools/test.py \
#   --config-file configs/waymo/semseg-litept-small-v1m1.py \
#   --options save_path=exp/waymo/semseg-litept-small-v1m1 \
#             weight=checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth


CLIP_NAME="segment-10203656353524179475_7625_000_7645_000_with_camera_labels"

"$PY" tools/test.py \
  --config-file configs/waymo/semseg-litept-small-v1m1.py \
  --options save_path=exp/waymo/single_clip_eval \
            weight=checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth \
            data.val.split="validation/${CLIP_NAME}"