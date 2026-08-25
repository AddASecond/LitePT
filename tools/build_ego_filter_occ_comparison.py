#!/usr/bin/env python3
"""Build a same-frame OCC baseline without ego filtering for viewer comparison."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from gridfs import GridFSBucket
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OCC = load_module("ego_compare_occ", "tools/robotruck_occupancy.py")
SAG = load_module("ego_compare_static", "tools/robotruck_static_agg.py")
INFER = load_module("ego_compare_infer", "tools/infer_robotruck_mongo_frame.py")
STORE = load_module("ego_compare_store", "tools/store_robotruck_occ_gridfs.py")


def ref(uri: str, dtype: str, shape) -> dict:
    return {"uri": uri, "dtype": dtype, "shape": list(shape), "byte_order": "little"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raw-cache", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--write-mongo", action="store_true")
    ap.add_argument("--mongo-uri", default=os.environ.get("ROBOTRUCK_MONGO_URI", "mongodb://krk030-mongodb:27017/?authSource=perception_experiment"))
    args = ap.parse_args()
    scene, raw, pred_dir = map(lambda p: Path(p).resolve(), (args.scene, args.raw_cache, args.pred_dir))
    index_path = scene / "index.json"
    index = json.loads(index_path.read_text())
    timestamps = [str(x.get("timestamp") or x["frame_id"]) for x in index["frames"]]
    cache = ROOT / "exp/robotruck/clip_video" / scene.name / "static_agg" / "static_voxel0.25_s5_no_ego_filter.npz"
    static = SAG.load_or_build_static_aggregate(
        raw, pred_dir, timestamps,
        load_lidar_bin=INFER.load_lidar_bin, lidar_cols=len(INFER.LIDAR_COLS),
        voxel=0.25, cache_path=cache, use_oracle_boxes=True,
        ego_filter={"enabled": False, "variant": "before_ego_filter/v1"},
    )
    raw_clip = json.loads((raw / "clip.json").read_text())
    clip_id = raw_clip["clip_id"]
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    if args.write_mongo:
        client.admin.command("ping")
    db = client.perception_experiment
    bucket = GridFSBucket(db, bucket_name="occ_blobs")
    files = db["occ_blobs.files"]

    for n, entry in enumerate(index["frames"], 1):
        ts = str(entry.get("timestamp") or entry["frame_id"])
        frame_dir = scene / "frames" / ts
        meta_path = frame_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        raw_meta = json.loads((raw / "frames" / ts / "frame.json").read_text())
        points = INFER.load_lidar_bin(raw / "frames" / ts / "lidar_merge.bin", num_cols=len(INFER.LIDAR_COLS))
        xyz, lidar = points[:, :3].astype(np.float32), points[:, 6].astype(np.int32)
        labels = np.load(pred_dir / f"{ts}_pred.npy").astype(np.int32)
        pose = ((raw_meta.get("dependency") or {}).get("ego_pose") or {}).get("pose")
        xyz_s, lab_s, lid_s = SAG.static_in_vehicle(
            static, SAG.ego_pose_to_T_map_vehicle(pose),
            x_range=(meta["grid"]["x_range"][0] * 1.5, meta["grid"]["x_range"][1] * 1.5),
            y_range=tuple(meta["grid"]["y_range"]),
            z_range=(meta["grid"]["z_range"][0] - 2, meta["grid"]["z_range"][1] + 5),
        )
        merged_xyz, merged_lab, _, _ = SAG.merge_static_dynamic(xyz_s, lab_s, lid_s, xyz, labels, lidar)
        grid = OCC.build_occupancy(
            merged_xyz, merged_lab,
            x_range=tuple(meta["grid"]["x_range"]), y_range=tuple(meta["grid"]["y_range"]),
            z_range=tuple(meta["grid"]["z_range"]), voxel=float(meta["grid"]["voxel"]), min_points=1,
        )
        prefix = f"frames/{ts}"
        arrays = {
            "ijk": ("occ_before_ego_ijk.i32.bin", grid.ijk.astype("<i4"), "int32"),
            "labels": ("occ_before_ego_labels.u8.bin", grid.labels.astype("u1"), "uint8"),
            "centers": ("occ_before_ego_centers.f32.bin", grid.centers.astype("<f4"), "float32"),
            "counts": ("occ_before_ego_counts.i32.bin", grid.counts.astype("<i4"), "int32"),
        }
        refs = {"n": int(len(grid.ijk))}
        for key, (name, array, dtype) in arrays.items():
            array.tofile(frame_dir / name)
            refs[key] = ref(f"{prefix}/{name}", dtype, array.shape)
        variants = meta["assets"].setdefault("occupancy_variants", {})
        variants["before_ego_filter"] = refs
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        entry.setdefault("n_occ_variants", {})["before_ego_filter"] = refs["n"]
        if args.write_mongo:
            mongo_assets = {}
            for key, asset in refs.items():
                if not isinstance(asset, dict):
                    continue
                mongo_assets[key] = STORE.store(
                    bucket, files, scene / asset["uri"],
                    f"occ-compare/lidar14_0813/{clip_id}/{ts}/before_ego_filter/{Path(asset['uri']).name}",
                    {"clip_id": clip_id, "timestamp": int(ts), "algorithm": "litept_before_ego_filter_v1", "asset": key},
                )
                mongo_assets[key].update({"dtype": asset["dtype"], "shape": asset["shape"]})
            db.occ_data_comparisons_lidar14_0813.update_one(
                {"clip_id": clip_id, "timestamp": int(ts)},
                {"$set": {"variants.before_ego_filter": {"algorithm": "litept_before_ego_filter/v1", "assets": mongo_assets}, "stats.before_ego_filter_n_occ": refs["n"]}},
            )
        if n % 10 == 0 or n == len(timestamps):
            print(f"built {n}/{len(timestamps)}", flush=True)
    index["occupancy_variants"] = {
        "default": "litept",
        "variants": [
            {"id": "before_ego_filter", "name": "Before ego filter (OCC)"},
            {"id": "litept", "name": "After ego filter (OCC)"},
            {"id": "tracking", "name": "OD/tracking boxes (OCC)"},
        ],
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
