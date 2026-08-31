#!/usr/bin/env python3
"""Fix index.json + retry store for scenes already exported by run_random10_inproc.py.

Why: earlier versions of the runner wrote index frames without `timestamp` / `raw_md5`.
store_robotruck_occ_gridfs needs one of those to map scene frames back to mongo raw frames.
This script does NOT re-run CUDA inference — it only patches index.json and calls store.
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import importlib.util
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
    sys.modules[name] = mod; return mod
store_mod = _load("store_robotruck_occ_gridfs", "tools/store_robotruck_occ_gridfs.py")


def patch_index(scene_dir: Path, backup_dir: Path) -> tuple[int, int]:
    index_path = scene_dir / "index.json"
    if not index_path.is_file():
        return 0, 0
    doc = json.loads(index_path.read_text())
    clip_id = doc.get("clip_id")
    frames_backup = backup_dir / str(clip_id) / "frames" if clip_id else None
    changed, total = 0, 0
    for f in doc.get("frames") or []:
        total += 1
        ts = str(f.get("timestamp") or f.get("frame_id") or f.get("ts") or "")
        f["timestamp"] = ts; f["frame_id"] = ts
        if not f.get("path"): f["path"] = f"frames/{ts}"
        if f.get("raw_md5"):
            continue
        # try reading from backup/frames/<ts>/frame.json
        fj = None
        candidates = []
        if frames_backup: candidates.append(frames_backup / ts / "frame.json")
        # also try clip_id inside scenes (run_random10 sometimes makes different dirs)
        for extra in list(backup_dir.glob(f"*/frames/{ts}/frame.json")):
            candidates.append(extra)
        for c in candidates:
            if c.is_file():
                try:
                    fj = json.loads(c.read_text()); break
                except Exception:
                    fj = None
        if fj is None:
            continue
        sensors = ((fj.get("dependency") or {}).get("sensors") or {})
        raw_md5 = None
        for key in ("lidar_merge_deskew", "lidar_merge", "lidar_merge_nodeskew"):
            md5 = (sensors.get(key) or {}).get("md5")
            if md5 and len(str(md5)) == 32:
                raw_md5 = str(md5); break
        if not raw_md5 and fj.get("md5") and len(str(fj["md5"])) == 32:
            raw_md5 = str(fj["md5"])
        if raw_md5:
            f["raw_md5"] = raw_md5
            changed += 1
    index_path.write_text(json.dumps(doc, indent=2))
    return changed, total


def call_store(scene, backup_root, asset_root, raw_frame_col, raw_clip_col, write):
    _sys_argv_save = sys.argv[:]
    try:
        sys.argv = [
            "store_robotruck_occ_gridfs.py",
            "--scene", str(Path(scene).resolve()),
            "--raw-frame-collection", raw_frame_col,
            *(["--raw-clip-collection", raw_clip_col] if raw_clip_col else []),
            "--asset-root", str(Path(asset_root).resolve()),
            "--backup-root", str(Path(backup_root).resolve()),
        ] + (["--write"] if write else [])
        return int(store_mod.main() or 0)
    finally:
        sys.argv[:] = _sys_argv_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", type=Path, required=True,
                    help="parent of scene dirs (batch_s5_<clip_id>), e.g. exp/robotruck/occ_scenes")
    ap.add_argument("--backup-root", type=Path, required=True,
                    help="materialized backup, with <backup>/<clip_id>/frames/<ts>/frame.json")
    ap.add_argument("--asset-root", type=Path, required=True)
    ap.add_argument("--raw-frame-collection", default="raw_data_frames_lidar14_0813")
    ap.add_argument("--raw-clip-collection", default="")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    scenes = sorted(p for p in args.scenes_root.iterdir() if p.is_dir() and (p/"index.json").is_file())
    print(f"scenes={len(scenes)}")
    results = []
    for sc in scenes:
        changed, total = patch_index(sc, args.backup_root)
        print(f"[{sc.name}] frames patched={changed}/{total}", end="")
        if changed == 0 and total == 0:
            print(" SKIP (empty)")
            continue
        rc = call_store(sc, args.backup_root, args.asset_root,
                        args.raw_frame_collection, args.raw_clip_collection, args.write)
        print(f" store_rc={rc}")
        results.append((sc.name, changed, total, rc))
    print("summary:")
    for n, c, t, r in results:
        print(f"  {n} patch {c}/{t} store_rc={r}")


if __name__ == "__main__":
    main()
