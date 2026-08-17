.venv_smoke/bin/python datasets/preprocessing/waymo/preprocess_waymo.py \
  --dataset_root /home/dev/02dataset/waymo_raw \
  --output_root ./data/waymo \
  --splits validation \
  --num_workers 16