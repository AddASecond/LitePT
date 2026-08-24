#!/usr/bin/env python3
"""Infer one raw Mongo frame and store versioned Occ artifacts in Mongo GridFS."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
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


INFER = load_module("robotruck_infer", "tools/infer_robotruck_mongo_frame.py")
OCC = load_module("robotruck_occ", "tools/robotruck_occupancy.py")
STORE = load_module("robotruck_gridfs", "tools/store_robotruck_occ_gridfs.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def put_array(bucket, files, array: np.ndarray, key: str, metadata: dict) -> dict:
    array = np.ascontiguousarray(array)
    payload = array.tobytes(order="C")
    digest = hashlib.sha256(payload).hexdigest()
    existing = files.find_one({"metadata.key": key, "metadata.sha256": digest})
    if existing:
        file_id = existing["_id"]
    else:
        for old in files.find({"metadata.key": key}, {"_id": 1}):
            bucket.delete(old["_id"])
        file_id = bucket.upload_from_stream(
            key,
            io.BytesIO(payload),
            metadata={
                **metadata,
                "key": key,
                "sha256": digest,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            },
        )
    return {
        "storage": "gridfs",
        "bucket": files.name.removesuffix(".files"),
        "gridfs_id": file_id,
        "filename": key,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "length": len(payload),
        "sha256": digest,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", required=True)
    ap.add_argument("--frame-md5", required=True)
    ap.add_argument("--dataset-version", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--mongo-uri", default=os.environ.get("ROBOTRUCK_MONGO_URI", "mongodb://krk030-mongodb:27017/?authSource=perception_experiment"))
    ap.add_argument("--bucket", default="occ_blobs")
    ap.add_argument("--config-file", default="configs/waymo/semseg-litept-small-v1m1.py")
    ap.add_argument("--weight", default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--occ-voxel", type=float, default=0.2)
    args = ap.parse_args()

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    raw = db[args.raw_frame_collection].find_one({"md5": args.frame_md5})
    if raw is None:
        raw = db[args.raw_frame_collection].find_one({"dependency.sensors.lidar_merge.md5": args.frame_md5})
    if raw is None:
        raise RuntimeError(f"raw frame md5={args.frame_md5} not found")
    lidar_md5 = INFER.md5_to_lidar_path.__name__ and (((raw.get("dependency") or {}).get("sensors") or {}).get("lidar_merge") or {}).get("md5") or raw.get("md5")
    lidar_path = STORE.resolve_raw(lidar_md5, "lidar")
    if lidar_path.suffix != ".bin":
        raise ValueError("single-frame test currently expects raw .bin; use production pipeline for PCD")
    raw_clip_id = raw.get("clip_id")
    raw_clip = db[args.raw_clip_collection].find_one({"clip_id": raw_clip_id})
    if raw_clip is None:
        raise RuntimeError(f"raw clip_id={raw_clip_id} not found")

    config_path, weight_path = ROOT / args.config_file, ROOT / args.weight
    checkpoint_sha = sha256_file(weight_path)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    params = {
        "grid_size": args.grid_size,
        "occ_voxel": args.occ_voxel,
        "x_range": [-30.0, 30.0], "y_range": [-200.0, 400.0], "z_range": [-5.0, 20.0],
    }
    fingerprint_payload = {
        "raw_collection": args.raw_frame_collection, "raw_md5": lidar_md5,
        "checkpoint_sha256": checkpoint_sha, "git_commit": git_commit, "params": params,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    now = datetime.now(timezone.utc)
    run_collection = db["occ_data_runs_" + args.raw_frame_collection.removeprefix("raw_data_frames_")]
    frame_collection = db["occ_data_frames_" + args.raw_frame_collection.removeprefix("raw_data_frames_")]
    clip_collection = db["occ_data_clips_" + args.raw_clip_collection.removeprefix("raw_data_clips_")]
    if run_collection.find_one({"run_id": args.run_id}):
        raise RuntimeError(f"run_id already exists: {args.run_id}")
    run_collection.create_index([("run_id", ASCENDING)], unique=True, name="run_id_unique")
    run_collection.insert_one({
        "schema_version": "litept_occ_run/v1", "dataset_version": args.dataset_version,
        "run_id": args.run_id, "artifact_fingerprint": fingerprint,
        "source": {"db": args.database, "frame_collection": args.raw_frame_collection, "clip_collection": args.raw_clip_collection, "frame_md5": lidar_md5, "clip_id": raw_clip_id},
        "model": {"name": "litept-small-waymo-semseg", "checkpoint": str(weight_path), "checkpoint_sha256": checkpoint_sha},
        "code": {"git_commit": git_commit}, "params": params,
        "status": {"state": "running", "expected_frames": 1, "completed_frames": 0, "failed_frames": 0},
        "created_at": now,
    })

    try:
        points = INFER.load_lidar_bin(lidar_path, num_cols=len(INFER.LIDAR_COLS))
        coord = points[:, :3].astype(np.float32)
        intensity = points[:, 3]
        strength = np.tanh(intensity.reshape(-1, 1) / 255.0).astype(np.float32)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, _ = INFER.load_segmentor(config_path, weight_path, device)
        pred = INFER.infer_frame(model, coord, strength, device, args.grid_size).astype(np.uint8)
        grid = OCC.build_occupancy(
            coord, pred, x_range=tuple(params["x_range"]), y_range=tuple(params["y_range"]),
            z_range=tuple(params["z_range"]), voxel=args.occ_voxel, min_points=1,
        )
        bucket, files = GridFSBucket(db, bucket_name=args.bucket), db[f"{args.bucket}.files"]
        prefix = f"{args.dataset_version}/{args.run_id}/frames/{lidar_md5}"
        metadata = {"dataset_version": args.dataset_version, "run_id": args.run_id, "artifact_fingerprint": fingerprint, "raw_frame_md5": lidar_md5}
        assets = {
            "occupancy": {
                "ijk": put_array(bucket, files, grid.ijk.astype(np.int32), f"{prefix}/occupancy/occ_ijk.i32.bin", metadata),
                "labels": put_array(bucket, files, grid.labels.astype(np.uint8), f"{prefix}/occupancy/occ_labels.u8.bin", metadata),
                "centers": put_array(bucket, files, grid.centers.astype(np.float32), f"{prefix}/occupancy/occ_centers.f32.bin", metadata),
                "counts": put_array(bucket, files, grid.counts.astype(np.int32), f"{prefix}/occupancy/occ_counts.i32.bin", metadata),
            },
            "points": {
                "xyz": put_array(bucket, files, coord.astype(np.float32), f"{prefix}/points/points_xyz.f32.bin", metadata),
                "labels": put_array(bucket, files, pred, f"{prefix}/points/points_labels.u8.bin", metadata),
                "lidar_id": put_array(bucket, files, points[:, 6].astype(np.uint8), f"{prefix}/points/points_lidar_id.u8.bin", metadata),
            },
        }
        cameras = STORE.raw_cameras(raw)
        frame_doc = {
            "schema_version": "litept_occ_frame/v3", "dataset_version": args.dataset_version, "run_id": args.run_id,
            "artifact_fingerprint": fingerprint, "md5": lidar_md5, "timestamp": raw.get("timestamp"),
            "clip_id": raw_clip_id, "bag_name": raw.get("bag_name"),
            "source": {"db": args.database, "frame_collection": args.raw_frame_collection, "clip_collection": args.raw_clip_collection, "raw_id": str(raw.get("_id")), "frame_md5": lidar_md5, "clip_id": raw_clip_id},
            "raw_assets": {"lidar_merge": {"md5": lidar_md5, "uri": str(lidar_path)}, "cameras": cameras},
            "model": {"name": "litept-small-waymo-semseg", "checkpoint_sha256": checkpoint_sha},
            "grid": {"voxel": args.occ_voxel, "origin": [params["x_range"][0], params["y_range"][0], params["z_range"][0]], "x_range": params["x_range"], "y_range": params["y_range"], "z_range": params["z_range"], "shape": list(grid.shape)},
            "stats": {"n_raw_points": int(coord.shape[0]), "n_occ": int(grid.ijk.shape[0])},
            "assets": {**assets, "cameras": cameras}, "ego_pose": ((raw.get("dependency") or {}).get("ego_pose")),
            "created_at": now, "updated_at": now,
        }
        frame_collection.create_index([("source.frame_collection", ASCENDING), ("source.frame_md5", ASCENDING), ("run_id", ASCENDING)], unique=True, name="source_frame_run_unique")
        frame_collection.insert_one(frame_doc)
        clip_collection.insert_one({
            "schema_version": "litept_occ_clip/v3", "dataset_version": args.dataset_version, "run_id": args.run_id,
            "artifact_fingerprint": fingerprint, "clip_id": raw_clip_id, "clip_name": raw_clip.get("clip_name"),
            "bag_name": raw_clip.get("bag_name"), "source": {"db": args.database, "clip_collection": args.raw_clip_collection, "frame_collection": args.raw_frame_collection, "raw_id": str(raw_clip.get("_id")), "clip_id": raw_clip_id},
            "frame_collection": frame_collection.name, "frame_count": 1, "status": {"state": "test-complete", "processed_frames": 1},
            "created_at": now, "updated_at": now,
        })
        run_collection.update_one({"run_id": args.run_id}, {"$set": {"status": {"state": "complete", "expected_frames": 1, "completed_frames": 1, "failed_frames": 0}, "completed_at": datetime.now(timezone.utc)}})
        print(json.dumps({"run_id": args.run_id, "dataset_version": args.dataset_version, "frame_collection": frame_collection.name, "clip_collection": clip_collection.name, "run_collection": run_collection.name, "gridfs_bucket": args.bucket, "raw_md5": lidar_md5, "n_raw_points": int(coord.shape[0]), "n_occ": int(grid.ijk.shape[0]), "artifact_fingerprint": fingerprint}, indent=2))
    except Exception as error:
        run_collection.update_one({"run_id": args.run_id}, {"$set": {"status.state": "failed", "status.failed_frames": 1, "error": f"{type(error).__name__}: {error}", "completed_at": datetime.now(timezone.utc)}})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
