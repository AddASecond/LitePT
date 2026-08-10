# LitePT Phase-0 smoke (working)

## What works now

Convert **2** local lidar bins → LitePT `.npy` layout, run **Waymo LitePT-S** pretrained segmentation on GPU.

| Metric | Result |
|--------|--------|
| Frames | 2 |
| Disk used | ~2.7 MB under `data/waymo_smoke/` |
| Peak VRAM | **~0.6 GB** (well under 18 GB) |
| Output | `pred_segment.npy` per frame, logits `[N, 22]` |

## Run

```bash
cd ~/01develop/LitePT
# use the smoke venv (system torch 2.8 + B200-compatible builds)
./scripts/run_smoke_seg.sh
# or:
.venv_smoke/bin/python tools/convert_few_clouds_for_smoke.py --no-hf-demo \
  --from-bin /path/to/a.bin --max-points 120000
.venv_smoke/bin/python tools/smoke_seg_infer.py \
  --data-root data/waymo_smoke --device cuda --no-amp --save-pred
```

## Env notes (this host)

- Use **`.venv_smoke`** (system PyTorch 2.8) — not `.venv` (torch 2.6 lacks sm_100 / B200).
- `numpy==1.26.4` pinned in `.venv_smoke`.
- Smoke forces **`spconv.ConvAlgo.Native`** (ImplicitGemm fails on B200 / HAMI `total_memory=0`).
- `flash_attn` uses SDPA fallback if missing.
- HAMI may report `total_mem_GB=0.0`; inference still runs.

## Not done yet

- Full Mongo `waymo_trainval_frames` export
- Real Waymo `segment.npy` GT (smoke uses ignore labels; preds are from pretrained ckpt)
- Training loop / 18g train config
