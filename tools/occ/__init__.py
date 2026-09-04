"""OCC lanes.

  prod   Mongo → quality_gate → export_scene → store → pipeline/batch[/inproc] → OCC
  infer  warmup (+ infer_robotruck_mongo_frame); export --reuse-pred
  debug  pose_badcase (T_MED/DRIFT/SOFT/HARD offline REJECT) → triage / scene_video

Shared: paths, occupancy, static_agg, cuda_env, gss_mongo, layer_scan.
Delivery gate (export): quality_gate. Offline reject list: pose_badcase (unchanged).
inproc: multi-clip HAMI runner; ego/occ defaults differ from export_scene CLI (see inproc).
"""
