# OCC lane contracts (A / B / C) — no shared business code A↔B

- **A**: `tools/warmup_robotruck_preds.py`, infer_*, clip_video — model owns preds  
- **B**: `tools/occ/*` — calls A only via CLI (`paths.ensure_preds`); never `import infer_*`  
- **C**: `occ_viewer` / `occ_qa` — only B-before (raw) or B-after (scene package) inputs  

**Pred:** `{pred_dir}/{ts}_pred.npy` int32 length = lidar N (7-col float32 bin).  
**Scene:** `robotruck_occ_scene/v1` — see SCHEMA.md.  
**A CLI:** `python tools/warmup_robotruck_preds.py --clip-dir … --pred-dir …`
