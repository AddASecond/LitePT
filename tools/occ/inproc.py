#!/usr/bin/env python3
"""Prod: multi-clip in-process export+store via export_scene (C4 thin wrapper).

Preserved defaults (≠ export_scene CLI): ego x±3.6 y∈[-1.2,1.2], occ_min_points=2, BEV±50.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from paths import ensure_import_path

ensure_import_path()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-json", required=True, type=Path)
    ap.add_argument("--backup-root", required=True, type=Path)
    ap.add_argument("--scenes-root", required=True, type=Path)
    ap.add_argument("--asset-root", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames-per-clip", type=int, default=0)
    ap.add_argument("--aggregate-static", action="store_true", default=True)
    ap.add_argument("--no-aggregate-static", dest="aggregate_static", action="store_false")
    ap.add_argument("--export-points", action="store_true", default=True)
    ap.add_argument("--occ-voxel", type=float, default=0.2)
    ap.add_argument("--occ-min-points", type=int, default=2)
    ap.add_argument("--max-export-points", type=int, default=65536)
    ap.add_argument("--raw-frame-collection", default="raw_data_frames_lidar14_0813")
    ap.add_argument("--raw-clip-collection", default="raw_data_clips_lidar14_0813")
    ap.add_argument("--write-store", action="store_true", default=True)
    ap.add_argument("--no-write-store", dest="write_store", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--geometry-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--quality-sample-frames", type=int, default=5)
    ap.add_argument("--layer-threshold", type=float, default=None)
    ap.add_argument("--pose-shift-threshold", type=float, default=None)
    args = ap.parse_args()

    import export_scene as export_mod
    import store as store_mod

    clips = json.loads(args.clips_json.read_text())
    status = []
    for i, c in enumerate(clips, 1):
        clip_id = c["clip_id"] if isinstance(c, dict) else str(c)
        tag = c.get("tag", "") if isinstance(c, dict) else ""
        scene_name = f"batch_s5_{clip_id}"
        print(f"\n==== [{i}/{len(clips)}] {clip_id} tag={tag} ====")
        if args.dry_run:
            status.append({"i": i, "clip_id": clip_id, "rc": 0, "msg": "dry-run"})
            continue
        # Materialized cache / scene name uses batch_s5_* prefix (legacy inproc).
        payload = [{"clip_id": scene_name}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name
        argv = [
            "--clips-json",
            tmp_path,
            "--backup-root",
            str(args.backup_root),
            "--out-dir",
            str(args.scenes_root),
            "--reuse-pred",
            "--stride",
            str(args.stride),
            "--max-frames",
            str(args.max_frames_per_clip),
            "--occ-voxel",
            str(args.occ_voxel),
            "--occ-min-points",
            str(args.occ_min_points),
            "--max-export-points",
            str(args.max_export_points),
            "--bev-x-half",
            "50",
            "--bev-y-min",
            "-50",
            "--bev-y-max",
            "50",
            "--z-min",
            "-3",
            "--z-max",
            "10",
            "--ego-x-range",
            "-3.6",
            "3.6",
            "--ego-y-range",
            "-1.2",
            "1.2",
            "--grid-size",
            "0.1",
            "--agg-stride",
            "1",
            "--quality-sample-frames",
            str(args.quality_sample_frames),
        ]
        if args.aggregate_static:
            argv.append("--aggregate-static")
        else:
            argv.append("--no-aggregate-static")
        if args.export_points:
            argv.append("--export-points")
        else:
            argv.append("--no-export-points")
        if args.geometry_quality_gate:
            argv.append("--geometry-quality-gate")
        else:
            argv.append("--no-geometry-quality-gate")
        if args.layer_threshold is not None:
            argv.extend(["--layer-threshold", str(args.layer_threshold)])
        if args.pose_shift_threshold is not None:
            argv.extend(["--pose-shift-threshold", str(args.pose_shift_threshold)])
        try:
            rc = int(export_mod.main(argv) or 0)
        except Exception as exc:
            print(f"  EXPORT FAIL: {exc}")
            status.append({"i": i, "clip_id": clip_id, "rc": 1, "msg": str(exc)})
            continue
        out_scene = (args.scenes_root / scene_name).resolve()
        rc_store = 0
        if args.write_store and rc == 0:
            try:
                rc_store = int(
                    store_mod.run(
                        scene=out_scene,
                        raw_frame_collection=args.raw_frame_collection,
                        raw_clip_collection=args.raw_clip_collection,
                        asset_root=Path(args.asset_root).resolve(),
                        backup_root=Path(args.backup_root).resolve(),
                        write=True,
                    )
                    or 0
                )
            except Exception as exc:
                print(f"  STORE FAIL: {exc}")
                rc_store = 99
        status.append(
            {
                "i": i,
                "clip_id": clip_id,
                "tag": tag,
                "rc": rc_store if rc == 0 else rc,
            }
        )

    print("\n============== SUMMARY ==============")
    for s in status:
        print(
            f"  [{s.get('i', '?'):>2}] rc={s.get('rc', '?')} {s.get('clip_id', '')} "
            f"tag={s.get('tag', '')} {s.get('msg', '')}"
        )


if __name__ == "__main__":
    main()
