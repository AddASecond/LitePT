#!/usr/bin/env python3
"""Prod: materialize Mongo raw → export_scene → store (one clip). CUDA only when exporting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paths import ROOT, DEFAULT_URI, RAW_ROOTS, ensure_import_path

ensure_import_path()

import numpy as np
from pymongo import MongoClient

import store as STORE


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str, ensure_ascii=False, indent=2))


def pcd_binary_to_litept(source: Path, target: Path) -> None:
    """Convert binary PCD x/y/z/intensity/ring/timestamp/lidar_id to float32 Nx7."""
    with source.open("rb") as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD has no DATA line: {source}")
            text = line.decode("ascii", errors="strict").strip()
            if text and not text.startswith("#"):
                key, *values = text.split()
                header[key.upper()] = values
            if text.upper().startswith("DATA "):
                break
        if header.get("DATA") != ["binary"]:
            raise ValueError(f"only DATA binary PCD is supported: {source}")
        fields = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        if any(count != 1 for count in counts):
            raise ValueError(f"PCD COUNT != 1 is unsupported: {source}")
        mapping = {
            ("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1",
            ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
            ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
        }
        dtype = np.dtype([(name, mapping[(kind, size)]) for name, kind, size in zip(fields, types, sizes)])
        points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
        raw = stream.read()
    expected = points * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"PCD byte size mismatch: {source}: {len(raw)} != {expected}")
    data = np.frombuffer(raw, dtype=dtype, count=points)
    required = ("x", "y", "z")
    if any(name not in data.dtype.names for name in required):
        raise ValueError(f"PCD lacks xyz: {source}")
    output = np.zeros((points, 7), dtype=np.float32)
    output[:, 0] = data["x"]
    output[:, 1] = data["y"]
    output[:, 2] = data["z"]
    if "intensity" in data.dtype.names:
        output[:, 3] = data["intensity"]
    if "ring" in data.dtype.names:
        output[:, 4] = data["ring"]
    if "lidar_id" in data.dtype.names:
        output[:, 6] = data["lidar_id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(target)


def link_or_convert_lidar(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".bin":
        target.symlink_to(source)
    elif source.suffix == ".pcd":
        pcd_binary_to_litept(source, target)
    else:
        raise ValueError(source)


def materialize_raw_cache(db, frame_collection: str, clip_collection: str, clip_id: str, cache: Path) -> int:
    raw_clip = db[clip_collection].find_one({"clip_id": clip_id})
    if raw_clip is None:
        raise RuntimeError(f"clip_id={clip_id} absent from {clip_collection}")
    frames = list(db[frame_collection].find({"clip_id": clip_id}).sort("timestamp", 1))
    if not frames:
        raise RuntimeError(f"no frames for clip_id={clip_id} in {frame_collection}")
    index = []
    for number, raw in enumerate(frames, 1):
        ts = str(raw["timestamp"])
        md5 = STORE.raw_lidar_md5(raw) or raw.get("md5")
        lidar = STORE.resolve_raw(md5, "lidar")
        frame_dir = cache / "frames" / ts
        link_or_convert_lidar(lidar, frame_dir / "lidar_merge.bin")
        sensors = ((raw.get("dependency") or {}).get("sensors") or {})
        copied_cameras = {}
        for name, sensor in sensors.items():
            if not name.startswith("camera") or not isinstance(sensor, dict):
                continue
            cam_md5 = sensor.get("md5")
            if not isinstance(cam_md5, str) or not STORE.MD5_RE.fullmatch(cam_md5):
                continue
            camera = STORE.resolve_raw(cam_md5, "camera")
            target = frame_dir / f"{name}.jpg"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(camera)
            copied_cameras[name] = target.name
        json_write(frame_dir / "frame.json", raw)
        index.append({
            "timestamp": raw["timestamp"], "md5": md5, "has_lidar": True,
            "n_cameras": len(copied_cameras),
            "copied": {"lidar_merge": "lidar_merge.bin", "cameras": copied_cameras},
        })
        if number % 50 == 0 or number == len(frames):
            print(f"materialized raw {number}/{len(frames)}", flush=True)
    json_write(cache / "clip.json", raw_clip)
    json_write(cache / "frames_index.json", index)
    json_write(cache / "SUMMARY.json", {
        "clip_id": clip_id, "mongo_frame_count": len(frames),
        "raw_roots": [str(p) for p in RAW_ROOTS],
    })
    return len(frames)


def _export_scene(argv: list[str]) -> int:
    from cuda_env import setup_cuda_env
    setup_cuda_env()
    import export_scene
    return int(export_scene.main(argv) or 0)


def run(
    *,
    raw_frame_collection: str,
    clip_id: str,
    raw_clip_collection: str = "",
    scene_name: str = "",
    database: str = "perception_experiment",
    mongo_uri: str = DEFAULT_URI,
    cache_root: str = "exp/robotruck/raw_volume_cache",
    scene_root: str = "exp/robotruck/occ_scenes",
    asset_root: str = "/data/rawdata-4/occupancy",
    stride: int = 1,
    max_frames: int = 0,
    force_infer: bool = False,
    write: bool = False,
) -> int:
    raw_clip_collection = raw_clip_collection or STORE.infer_collection(
        raw_frame_collection, "frames", "clips"
    )
    scene_name = scene_name or f"mongo_{raw_frame_collection}_{clip_id}"
    cache = (ROOT / cache_root / scene_name).resolve()
    scene = (ROOT / scene_root / scene_name).resolve()
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[database]
    if force_infer or not (scene / "index.json").is_file():
        materialize_raw_cache(db, raw_frame_collection, raw_clip_collection, clip_id, cache)
        rc = _export_scene([
            "--clip", scene_name,
            "--backup-root", cache_root,
            "--out-dir", scene_root,
            "--stride", str(stride),
            "--max-frames", str(max_frames),
            "--export-points",
            "--aggregate-static",
        ])
        if rc:
            return rc
    else:
        print(f"reuse existing inference scene: {scene}", flush=True)
    return STORE.run(
        scene=scene,
        raw_frame_collection=raw_frame_collection,
        raw_clip_collection=raw_clip_collection,
        backup_root=cache_root,
        asset_root=asset_root,
        write=write,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", default="")
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--scene-name", default="")
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--mongo-uri", default=DEFAULT_URI)
    ap.add_argument("--cache-root", default="exp/robotruck/raw_volume_cache")
    ap.add_argument("--scene-root", default="exp/robotruck/occ_scenes")
    ap.add_argument("--asset-root", default="/data/rawdata-4/occupancy")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--force-infer", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    return run(
        raw_frame_collection=args.raw_frame_collection,
        raw_clip_collection=args.raw_clip_collection,
        clip_id=args.clip_id,
        scene_name=args.scene_name,
        database=args.database,
        mongo_uri=args.mongo_uri,
        cache_root=args.cache_root,
        scene_root=args.scene_root,
        asset_root=args.asset_root,
        stride=args.stride,
        max_frames=args.max_frames,
        force_infer=args.force_infer,
        write=args.write,
    )


if __name__ == "__main__":
    raise SystemExit(main())
