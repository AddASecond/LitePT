"""Robotruck occupancy delivery toolkit.

Layout (by goal):
  Production:  quality_gate, occupancy, static_agg, export_scene, store, ingest,
               pipeline, batch_lidar14, run_inproc, pred_warmup, gss_mongo
  Debug/QA:    pose_badcase, layer_scan, triage_viz, validate_projection,
               scene_video  (+ tools/occ_viewer)
"""
