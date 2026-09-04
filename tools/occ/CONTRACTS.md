# OCC lane contracts (A / B / C) — no shared business code A↔B

- **A**: `tools/warmup_robotruck_preds.py`, infer_*, clip_video — model owns preds  
- **B**: `tools/occ/*` — calls A only via CLI (`paths.ensure_preds`); never `import infer_*`  
- **C viewer**: standalone repo `/home/dev/01develop/occ_viewer` — scene packages only  
- **C QA**: `tools/occ_qa` — B-before raw / Mongo (still in LitePT for now)  

**Pred:** `{pred_dir}/{ts}_pred.npy` int32 length = lidar N (7-col float32 bin).  
**Scene:** `robotruck_occ_scene/v1` — see `/home/dev/01develop/occ_viewer/SCHEMA.md`.  
**A CLI:** `python tools/warmup_robotruck_preds.py --clip-dir … --pred-dir …`  
**Viewer:** `cd /home/dev/01develop/occ_viewer && python serve.py --scene <scene_dir>`
