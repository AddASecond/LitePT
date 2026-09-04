"""Robotruck occupancy toolkit.

Public entrypoints (use these):
  produce.py   delivery: export | pipeline | batch | inproc | warmup | store
  qa.py        offline QA: scan | triage | layer | project
  scene_video.py
  occ_viewer/  interactive debug viewer

Libraries (imported, not run):
  quality_gate, occupancy, static_agg, gss_mongo, export_scene, store

Internals live in _impl/ (do not call directly unless debugging).
"""
