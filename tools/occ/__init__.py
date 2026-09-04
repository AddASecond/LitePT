"""OCC delivery & QA package (importable modules, no command router).

Delivery:
  quality_gate → export_scene → store → pipeline / batch / inproc
QA / debug:
  pose_badcase, triage, scene_video, layer_scan, validate_projection
Libs:
  occupancy, static_agg, gss_mongo, cuda_env
"""
