#!/usr/bin/env python3
"""Add tracking-aware OCC variants to one exported scene and MongoDB."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gridfs import GridFSBucket
from pymongo import ASCENDING, MongoClient

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OCC = load_module("tracking_compare_occ", "tools/robotruck_occupancy.py")
SAG = load_module("tracking_compare_static", "tools/robotruck_static_agg.py")
INFER = load_module("tracking_compare_infer", "tools/infer_robotruck_mongo_frame.py")
STORE = load_module("tracking_compare_store", "tools/store_robotruck_occ_gridfs.py")


def read_array(scene: Path, ref: dict, dtype) -> np.ndarray:
    value = np.fromfile(scene / ref["uri"], dtype=dtype)
    return value.reshape(ref["shape"])


def asset_ref(uri: str, dtype: str, shape: list[int]) -> dict:
    return {"uri": uri, "dtype": dtype, "shape": shape, "byte_order": "little"}


def points_in_tracking_boxes(
    xyz: np.ndarray, objects: list[dict], margin: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Return inside-any-box mask and per-point track ordinal (0 means none)."""
    inside_any = np.zeros(len(xyz), dtype=bool)
    track = np.zeros(len(xyz), dtype=np.int32)
    for ordinal, obj in enumerate(objects, 1):
        center, size = obj.get("center_imu"), obj.get("size")
        if not center or not size or len(center) < 3 or len(size) < 3:
            continue
        dx = xyz[:, 0] - float(center[0])
        dy = xyz[:, 1] - float(center[1])
        dz = xyz[:, 2] - float(center[2])
        yaw = float(obj.get("orientation_imu") or 0.0)
        longitudinal = np.cos(yaw) * dx + np.sin(yaw) * dy
        lateral = -np.sin(yaw) * dx + np.cos(yaw) * dy
        length, width, height = map(float, size[:3])
        mask = (
            (np.abs(longitudinal) <= length * 0.5 + margin)
            & (np.abs(lateral) <= width * 0.5 + margin)
            & (np.abs(dz) <= height * 0.5 + margin)
        )
        inside_any |= mask
        track[(track == 0) & mask] = ordinal
    return inside_any, track


def write_variant(frame_dir: Path, grid) -> dict:
    prefix = f"frames/{frame_dir.name}"
    files = {
        "ijk": ("occ_tracking_ijk.i32.bin", grid.ijk.astype("<i4"), "int32"),
        "labels": ("occ_tracking_labels.u8.bin", grid.labels.astype("u1"), "uint8"),
        "centers": ("occ_tracking_centers.f32.bin", grid.centers.astype("<f4"), "float32"),
        "counts": ("occ_tracking_counts.i32.bin", grid.counts.astype("<i4"), "int32"),
    }
    refs = {"n": int(len(grid.ijk))}
    for key, (name, array, dtype) in files.items():
        array.tofile(frame_dir / name)
        refs[key] = asset_ref(f"{prefix}/{name}", dtype, list(array.shape))
    return refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raw-cache", required=True)
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--mongo-uri", default=os.environ.get(
        "ROBOTRUCK_MONGO_URI",
        "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
    ))
    ap.add_argument("--bucket", default="occ_blobs")
    ap.add_argument("--write-mongo", action="store_true")
    args = ap.parse_args()

    scene, raw_cache = Path(args.scene).resolve(), Path(args.raw_cache).resolve()
    index_path = scene / "index.json"
    index = json.loads(index_path.read_text())
    raw_clip = json.loads((raw_cache / "clip.json").read_text())
    clip_id = raw_clip.get("clip_id")
    if not clip_id:
        raise ValueError(f"raw cache clip.json has no clip_id: {raw_cache}")
    static = index.get("static_agg") or {}
    xyz_map = read_array(scene, static["xyz_map"], np.float32)
    static_labels = read_array(scene, static["labels"], np.uint8).reshape(-1)
    static_lidar = read_array(scene, static["lidar_id"], np.uint8).reshape(-1)
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    bucket = GridFSBucket(db, bucket_name=args.bucket)
    files = db[f"{args.bucket}.files"]
    comparisons = db.occ_data_comparisons_lidar14_0813
    comparisons.create_index(
        [("clip_id", ASCENDING), ("timestamp", ASCENDING)], unique=True,
        name="clip_timestamp_unique",
    )
    comparison_stats = []

    for number, entry in enumerate(index["frames"], 1):
        ts = str(entry.get("timestamp") or entry["frame_id"])
        frame_dir = scene / "frames" / ts
        meta_path = frame_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        raw_meta = json.loads((raw_cache / "frames" / ts / "frame.json").read_text())
        points = INFER.load_lidar_bin(
            raw_cache / "frames" / ts / "lidar_merge.bin", num_cols=len(INFER.LIDAR_COLS)
        )
        xyz, lidar_ids = points[:, :3].astype(np.float32), points[:, 6].astype(np.uint8)
        pred_path = ROOT / "exp/robotruck/clip_video" / scene.name / "preds" / f"{ts}_pred.npy"
        labels = np.load(pred_path).astype(np.uint8)
        tracking_objects = (((raw_meta.get("dependency") or {}).get("lidar_objects") or {}).get("objects") or [])
        dynamic_mask, track_ordinal = points_in_tracking_boxes(xyz, tracking_objects)
        pose = ((raw_meta.get("dependency") or {}).get("ego_pose") or {}).get("pose")
        static_agg = {"xyz_map": xyz_map, "labels": static_labels, "lidar_ids": static_lidar}
        xyz_static, lab_static, _ = SAG.static_in_vehicle(
            static_agg,
            SAG.ego_pose_to_T_map_vehicle(pose),
            x_range=(meta["grid"]["x_range"][0] * 1.5, meta["grid"]["x_range"][1] * 1.5),
            y_range=tuple(meta["grid"]["y_range"]),
            z_range=(meta["grid"]["z_range"][0] - 2.0, meta["grid"]["z_range"][1] + 5.0),
        )
        merged_xyz = np.concatenate([xyz_static, xyz[dynamic_mask]], axis=0)
        merged_labels = np.concatenate([lab_static, labels[dynamic_mask]], axis=0)
        grid = OCC.build_occupancy(
            merged_xyz,
            merged_labels,
            x_range=tuple(meta["grid"]["x_range"]),
            y_range=tuple(meta["grid"]["y_range"]),
            z_range=tuple(meta["grid"]["z_range"]),
            voxel=float(meta["grid"]["voxel"]),
            min_points=1,
        )
        tracking_refs = write_variant(frame_dir, grid)
        baseline = meta["assets"]["occupancy"]
        meta["assets"]["occupancy_variants"] = {
            "litept": baseline,
            "tracking": tracking_refs,
        }
        meta["comparison"] = {
            "default_variant": "litept",
            "variants": {
                "litept": {"dynamic_source": "LitePT semantic dynamic classes"},
                "tracking": {"dynamic_source": "MongoDB dependency.lidar_objects oriented boxes"},
            },
            "tracking_object_count": len(tracking_objects),
            "tracking_dynamic_point_count": int(dynamic_mask.sum()),
            "tracked_point_count": int((track_ordinal > 0).sum()),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        entry["n_occ_variants"] = {"litept": baseline["n"], "tracking": tracking_refs["n"]}

        if args.write_mongo:
            mongo_assets = {}
            source = {"clip_id": clip_id, "timestamp": int(ts), "algorithm": "tracking_boxes_v1"}
            for key, ref in tracking_refs.items():
                if not isinstance(ref, dict):
                    continue
                mongo_assets[key] = STORE.store(
                    bucket, files, scene / ref["uri"],
                    f"occ-compare/lidar14_0813/{clip_id}/{ts}/tracking/{Path(ref['uri']).name}",
                    {**source, "asset": key, "dtype": ref["dtype"], "shape": ref["shape"]},
                )
                mongo_assets[key].update({"dtype": ref["dtype"], "shape": ref["shape"]})
            baseline_doc = db.occ_data_frames_lidar14_0813.find_one(
                {"source.clip_id": clip_id, "timestamp": int(ts)}, {"_id": 1, "assets.occupancy": 1}
            )
            comparisons.update_one(
                {"clip_id": clip_id, "timestamp": int(ts)},
                {"$set": {
                    "schema_version": "litept_occ_comparison/v1",
                    "clip_id": clip_id,
                    "timestamp": int(ts),
                    "source": {"raw_frame_collection": "raw_data_frames_lidar14_0813", "raw_md5": raw_meta.get("md5")},
                    "variants": {
                        "litept": {"frame_document_id": str(baseline_doc["_id"]) if baseline_doc else None, "assets": ((baseline_doc or {}).get("assets") or {}).get("occupancy")},
                        "tracking": {"algorithm": "oriented_lidar_object_boxes/v1", "assets": mongo_assets},
                    },
                    "stats": {"litept_n_occ": baseline["n"], "tracking_n_occ": tracking_refs["n"], "tracking_objects": len(tracking_objects), "tracking_dynamic_points": int(dynamic_mask.sum())},
                    "updated_at": datetime.now(timezone.utc),
                }, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                upsert=True,
            )
        comparison_stats.append({"timestamp": ts, "litept": baseline["n"], "tracking": tracking_refs["n"], "objects": len(tracking_objects), "dynamic_points": int(dynamic_mask.sum())})
        if number % 10 == 0 or number == len(index["frames"]):
            print(f"built {number}/{len(index['frames'])}", flush=True)

    index["occupancy_variants"] = {
        "default": "litept",
        "variants": [
            {"id": "litept", "name": "LitePT dynamic"},
            {"id": "tracking", "name": "OD/tracking boxes"},
        ],
    }
    index["comparison_summary"] = {
        "frames": len(comparison_stats),
        "mean_litept_n_occ": float(np.mean([x["litept"] for x in comparison_stats])),
        "mean_tracking_n_occ": float(np.mean([x["tracking"] for x in comparison_stats])),
        "mean_tracking_objects": float(np.mean([x["objects"] for x in comparison_stats])),
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(json.dumps(index["comparison_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
