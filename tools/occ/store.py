#!/usr/bin/env python3
"""Prod: write OCC scene packages into Mongo + content-addressed files under /data/rawdata*."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, MongoClient

from paths import ROOT, RAW_ROOTS, DEFAULT_ASSET_ROOT, DEFAULT_URI
MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def infer_collection(raw_name: str, source: str, target: str) -> str:
    marker = f"raw_data_{source}"
    if not raw_name.startswith(marker):
        raise ValueError(f"cannot infer from collection {raw_name!r}; expected prefix {marker!r}")
    return f"raw_data_{target}{raw_name[len(marker):]}"

def infer_occ_collection(raw_name: str) -> str:
    if not raw_name.startswith("raw_data_"):
        raise ValueError(f"raw collection must start with 'raw_data_': {raw_name}")
    return "occ_data_" + raw_name[len("raw_data_"):]

def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data

def raw_lidar_md5(raw: dict[str, Any]) -> str | None:
    sensors = ((raw.get("dependency") or {}).get("sensors") or {})
    for name in ("lidar_merge_deskew", "lidar_merge", "lidar_merge_nodeskew"):
        value = ((sensors.get(name) or {}).get("md5"))
        if isinstance(value, str) and MD5_RE.fullmatch(value):
            return value.lower()
    return None

def md5_from_backup(scene_frame: dict[str, Any], backup_root: Path | None) -> str | None:
    for value in (
        scene_frame.get("raw_md5"),
        scene_frame.get("md5"),
        scene_frame.get("frame_md5"),
    ):
        if isinstance(value, str) and MD5_RE.fullmatch(value):
            return value.lower()
    if backup_root is None:
        return None
    ts = str(scene_frame.get("timestamp") or scene_frame.get("frame_id") or "")
    frame_json = backup_root / "frames" / ts / "frame.json"
    if not frame_json.is_file():
        return None
    raw = load_json(frame_json)
    sensors = ((raw.get("dependency") or {}).get("sensors") or {})
    lidar_md5s = [
        ((sensors.get(name) or {}).get("md5"))
        for name in ("lidar_merge_deskew", "lidar_merge", "lidar_merge_nodeskew")
    ]
    for value in (*lidar_md5s, raw.get("md5")):
        if isinstance(value, str) and MD5_RE.fullmatch(value):
            return value.lower()
    return None

def find_raw_frame(collection, md5: str, timestamp: int | str | None = None) -> dict[str, Any] | None:
    if timestamp is not None:
        values: list[Any] = [timestamp]
        try:
            numeric = int(timestamp)
            if numeric != timestamp:
                values.append(numeric)
        except (TypeError, ValueError):
            pass
        raw = collection.find_one({"timestamp": {"$in": values}})
        if raw is not None and md5 in {
            raw.get("md5"), raw_lidar_md5(raw),
            ((((raw.get("dependency") or {}).get("sensors") or {}).get("lidar_merge_nodeskew") or {}).get("md5")),
        }:
            return raw
    return collection.find_one({
        "$or": [
            {"md5": md5},
            {"dependency.sensors.lidar_merge.md5": md5},
            {"dependency.sensors.lidar_merge_deskew.md5": md5},
            {"dependency.sensors.lidar_merge_nodeskew.md5": md5},
        ]
    })

def create_indexes(frame_collection, clip_collection) -> None:
    frame_collection.create_index(
        [("source.frame_collection", ASCENDING), ("source.frame_md5", ASCENDING), ("model.version", ASCENDING)],
        unique=True, name="source_frame_model_unique",
    )
    frame_collection.create_index([("bag_name", ASCENDING), ("timestamp", ASCENDING)], name="bag_timestamp")
    frame_collection.create_index([("md5", ASCENDING)], name="frame_md5")
    clip_collection.create_index(
        [("source.clip_collection", ASCENDING), ("source.clip_id", ASCENDING), ("model.version", ASCENDING)],
        unique=True, name="source_clip_model_unique",
    )
    clip_collection.create_index([("bag_name", ASCENDING)], name="bag_name")

def resolve_raw(md5: str, kind: str) -> Path:
    exts = (".bin", ".pcd") if kind == "lidar" else (".jpg", ".jpeg", ".png")
    for root in RAW_ROOTS:
        for ext in exts:
            path = root / kind / md5[:2] / md5[2:4] / f"{md5[4:]}{ext}"
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"raw {kind} md5={md5} absent from allowed rawdata roots")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def content_addressed_path(root: Path, digest: str, suffix: str) -> Path:
    return root / digest[:2] / digest[2:4] / f"{digest[4:]}{suffix}"

def store(path: Path, asset_root: Path) -> dict:
    """Copy an immutable blob once and return its Mongo-safe reference."""
    digest = sha256(path)
    suffix = "".join(path.suffixes) or ".bin"
    target = content_addressed_path(asset_root, digest, suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            copy_digest = hashlib.sha256()
            with path.open("rb") as src, temporary.open("wb") as dst:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    copy_digest.update(chunk)
                    dst.write(chunk)
            if copy_digest.hexdigest() != digest:
                raise IOError(f"checksum mismatch while storing {path}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "storage": "content_addressed_file", "uri": str(target),
        "length": target.stat().st_size, "sha256": digest,
    }

def upload_group(scene: Path, group: dict | None, asset_root: Path, *, publish=None):
    if group is None:
        return None
    output = {}
    for name, ref in group.items():
        if not isinstance(ref, dict) or "uri" not in ref:
            output[name] = ref
            continue
        uri = ref["uri"]
        path = (Path(uri) if uri.startswith("/") else scene / uri).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = {k: v for k, v in ref.items() if k != "uri"}
        output[name].update(publish(path) if publish else store(path, asset_root))
    return output

def gridfs_publisher(db, bucket_name: str):
    from gridfs import GridFSBucket
    if not re.fullmatch(r"[\w-]+", bucket_name):
        raise ValueError("invalid GridFS bucket name")
    bucket = GridFSBucket(db, bucket_name=bucket_name)
    def publish(path):
        digest = sha256(path)
        existing = db[f"{bucket_name}.files"].find_one({"metadata.sha256": digest})
        if existing:
            if existing["length"] != path.stat().st_size:
                raise ValueError("GridFS digest/length conflict")
            identifier = existing["_id"]
        else:
            with path.open("rb") as stream:
                identifier = bucket.upload_from_stream(path.name, stream, metadata={"sha256": digest})
        return {"storage": "gridfs", "bucket": bucket_name, "gridfs_id": str(identifier),
                "sha256": digest, "length": path.stat().st_size, "filename": path.name}
    return publish

def validate_camera_assets(meta, raw_cameras):
    import numpy as np
    cameras = (meta.get("assets") or {}).get("cameras") or []
    if raw_cameras and not cameras:
        raise ValueError("Scene has no calibrated cameras; re-export with current camera names")
    for camera in cameras:
        if (len(camera.get("K", [])) != 9 or len(camera.get("T_c_v_lidar_ref", [])) != 16
                or not camera.get("image", {}).get("uri")):
            raise ValueError(f"incomplete camera calibration: {camera.get('name')}")
        if not np.isfinite(camera['K']).all() or not np.isfinite(camera['T_c_v_lidar_ref']).all():
            raise ValueError(f"non-finite camera calibration: {camera.get('name')}")
    return cameras

def validate_scene_assets(scene, value):
    """Check every referenced file before the first Mongo/blob write."""
    import numpy as np
    if isinstance(value, list):
        for child in value:
            validate_scene_assets(scene, child)
    elif isinstance(value, dict):
        if 'uri' in value:
            path = (scene / value['uri']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if 'dtype' in value and 'shape' in value:
                expected = int(np.prod(value['shape'])) * np.dtype(value['dtype']).itemsize
                if path.stat().st_size != expected:
                    raise ValueError(f"asset shape/length mismatch: {path}")
        else:
            for child in value.values():
                validate_scene_assets(scene, child)


def published_ego_pose(meta, raw):
    from static_agg import lidar_reference_timestamp
    if (meta.get('pose_reference') != 'lidar_timestamp'
            or meta.get('ego_pose_timestamp_ns') != str(lidar_reference_timestamp(raw))):
        raise ValueError('Scene ego pose is not at the LiDAR reference time; re-export before publishing')
    pose = meta.get('ego_pose')
    if not pose or 'position' not in pose or 'orientation' not in pose:
        raise ValueError('Scene reference ego pose is missing')
    ts = int(meta['ego_pose_timestamp_ns'])
    return {'header': {'stamp': {'sec': ts // 1_000_000_000, 'nanosec': ts % 1_000_000_000},
                       'frame_id': 'map'}, 'pose': pose}

def raw_cameras(raw: dict) -> list[dict]:
    sensors = ((raw.get("dependency") or {}).get("sensors") or {})
    output = []
    for name, sensor in sorted(sensors.items()):
        if not name.startswith("camera") or not isinstance(sensor, dict):
            continue
        md5 = sensor.get("md5")
        if isinstance(md5, str) and MD5_RE.fullmatch(md5):
            output.append({
                "name": name, "md5": md5, "timestamp": sensor.get("timestamp"),
                "uri": str(resolve_raw(md5, "camera")),
            })
    return output

def run(
    *,
    scene: str | Path,
    raw_frame_collection: str,
    raw_clip_collection: str = "",
    occ_frame_collection: str = "",
    occ_clip_collection: str = "",
    database: str = "perception_experiment",
    mongo_uri: str = DEFAULT_URI,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    model_version: str = "v1",
    backup_root: str | Path = "data/robotruck_clips_backup",
    resume: bool = False,
    write: bool = False,
    split_tag: str = "",
    asset_storage: str = "file",
    gridfs_bucket: str = "occ_blobs",
    s3_bucket: str = "",
    s3_prefix: str = "occ_assets",
    s3_mount_root: str = "",
    s3_profile: str | None = None,
) -> int:
    if asset_storage not in ("file", "gridfs", "s3"):
        raise ValueError("asset_storage must be file, gridfs or s3")
    if asset_storage == 's3' and (not s3_bucket or not s3_mount_root or not Path(s3_mount_root).is_absolute()):
        raise ValueError('S3 bucket and absolute consumer mount root are required')
    if split_tag not in ("", "train", "val"):
        raise ValueError("split_tag must be train or val")
    scene = Path(scene).resolve()
    asset_root = Path(asset_root)
    index = load_json(scene / "index.json")
    raw_clip_name = raw_clip_collection or infer_collection(raw_frame_collection, "frames", "clips")
    occ_frame_name = occ_frame_collection or infer_occ_collection(raw_frame_collection)
    occ_clip_name = occ_clip_collection or infer_occ_collection(raw_clip_name)
    if len({raw_frame_collection, raw_clip_name, occ_frame_name, occ_clip_name}) != 4:
        raise ValueError("raw and OCC frame/clip collections must all be distinct")
    backup = (ROOT / backup_root / scene.name).resolve()
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[database]
    raw_frames, raw_clips = db[raw_frame_collection], db[raw_clip_name]
    occ_frames, occ_clips = db[occ_frame_name], db[occ_clip_name]

    prepared, clip_ids = [], set()
    for entry in index.get("frames") or []:
        md5 = md5_from_backup(entry, backup)
        raw = find_raw_frame(raw_frames, md5, entry.get("timestamp")) if md5 else None
        if raw is None:
            raise RuntimeError(f"raw trace failed for scene frame {entry.get('timestamp')}, md5={md5}")
        lidar_md5 = raw_lidar_md5(raw) or raw.get("md5")
        lidar_path = resolve_raw(lidar_md5, "lidar")
        cameras = raw_cameras(raw)
        clip_ids.add(raw.get("clip_id"))
        ts = str(entry.get("timestamp") or entry.get("frame_id"))
        meta = load_json(scene / (entry.get("meta_uri") or f"frames/{ts}/meta.json"))
        validate_camera_assets(meta, cameras)
        published_ego_pose(meta, raw)
        validate_scene_assets(scene, meta.get('assets') or {})
        prepared.append((entry, raw, meta, lidar_path, cameras))
    if len(clip_ids) != 1:
        raise RuntimeError(f"expected one raw clip, got {clip_ids}")
    clip_id = next(iter(clip_ids))
    raw_clip = raw_clips.find_one({"clip_id": clip_id})
    if raw_clip is None:
        raise RuntimeError(f"raw clip {clip_id} absent from {raw_clip_name}")
    validate_scene_assets(scene, index.get('static_agg'))

    print(json.dumps({
        "mode": f"write-{asset_storage}" if write else "dry-run", "scene": str(scene),
        "frames": len(prepared), "raw_roots": [str(p) for p in RAW_ROOTS],
        "raw": {"frames": raw_frame_collection, "clips": raw_clip_name, "clip_id": clip_id},
        "occ": {"frames": occ_frame_name, "clips": occ_clip_name,
                "asset_storage": asset_storage, "gridfs_bucket": gridfs_bucket,
                "asset_root": str(asset_root.resolve()) if asset_storage == 'file' else None},
    }, default=str, indent=2), flush=True)
    if not write:
        return 0

    create_indexes(occ_frames, occ_clips)
    publish = gridfs_publisher(db, gridfs_bucket) if asset_storage == "gridfs" else None
    if asset_storage == 's3':
        from s3_assets import s3_publisher
        publish = s3_publisher(s3_bucket, s3_prefix, s3_mount_root, profile=s3_profile)
        # Finish remote publication before exposing any frame metadata. Parallel
        # requests keep per-object latency from dominating a full clip upload.
        from concurrent.futures import ThreadPoolExecutor
        def paths_in(value):
            if isinstance(value, dict):
                if isinstance(value.get('uri'), str):
                    yield scene / value['uri']
                else:
                    for child in value.values():
                        yield from paths_in(child)
            elif isinstance(value, list):
                for child in value:
                    yield from paths_in(child)
        paths = sorted(set(paths_in([meta['assets'] for _, _, meta, _, _ in prepared]
                                    + [index.get('static_agg')])))
        references = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for number, (path, reference) in enumerate(zip(paths, pool.map(publish, paths)), 1):
                references[path] = reference
                if number % 500 == 0 or number == len(paths):
                    print(f'S3 uploaded {number}/{len(paths)}', flush=True)
        publish = lambda path: dict(references[path])
    now = utc_now()
    for number, (entry, raw, meta, lidar_path, cameras) in enumerate(prepared, 1):
        md5 = raw_lidar_md5(raw) or raw["md5"]
        identity = {
            "source.frame_collection": raw_frame_collection,
            "source.frame_md5": md5,
            "model.version": model_version,
        }
        if resume and occ_frames.find_one(
            {**identity, "schema_version": "litept_occ_frame/v2"}, {"_id": 1}
        ):
            continue
        source = {
            "db": database, "frame_collection": raw_frame_collection,
            "clip_collection": raw_clip_name, "raw_id": str(raw.get("_id")),
            "frame_md5": md5, "clip_id": clip_id,
        }
        current = meta.get("assets") or {}
        calibrated_cameras = []
        for camera in validate_camera_assets(meta, cameras):
            calibrated_cameras.append({**camera, "image": upload_group(
                scene, {"image": camera["image"]}, asset_root, publish=publish)["image"]})
        assets = {
            "occupancy": upload_group(scene, current.get("occupancy"), asset_root, publish=publish),
            "points": upload_group(scene, current.get("points"), asset_root, publish=publish),
            "frame_sensor_points": upload_group(scene, current.get("frame_sensor_points"), asset_root, publish=publish),
            "cameras": calibrated_cameras,
        }
        doc = {
            "schema_version": "litept_occ_frame/v2", "md5": md5,
            "timestamp": raw.get("timestamp", entry.get("timestamp")), "clip_id": clip_id,
            "bag_name": raw.get("bag_name"),
            "tag": [split_tag] if split_tag else sorted(set((raw.get("tag") or []) + ["litept", "occ", "semseg"])),
            "source": source,
            "raw_assets": {"lidar_merge": {"md5": md5, "uri": str(lidar_path)}, "cameras": cameras},
            "model": {"name": "litept-small-waymo-semseg", "version": model_version, "taxonomy": "waymo-22"},
            "coordinate": meta.get("coordinate"), "grid": meta.get("grid"), "stats": meta.get("stats"),
            "assets": assets, "ego_pose": published_ego_pose(meta, raw),
            "pose_reference": meta['pose_reference'],
            "ego_pose_timestamp_ns": meta['ego_pose_timestamp_ns'],
            "provenance": {
                "intermediate_scene": str(scene),
                "scene_schema_version": index.get("schema_version"),
                "geometry_quality": index.get("geometry_quality", index.get("pose_quality")),
                "source_ego_pose": ((raw.get('dependency') or {}).get('ego_pose')),
            },
            "updated_at": now,
        }
        occ_frames.update_one(
            identity, {"$set": doc, "$setOnInsert": {"created_at": now}}, upsert=True,
        )
        if number % 10 == 0 or number == len(prepared):
            print(f"uploaded {number}/{len(prepared)}", flush=True)

    static = upload_group(scene, index.get("static_agg"), asset_root, publish=publish)
    timestamps = [raw.get("timestamp") for _, raw, _, _, _ in prepared]
    clip_doc = {
        "schema_version": "litept_occ_clip/v2", "clip_id": clip_id,
        "clip_name": raw_clip.get("clip_name"), "bag_name": raw_clip.get("bag_name"),
        "bag_path": raw_clip.get("bag_path"),
        "tag": [split_tag] if split_tag else sorted(set((raw_clip.get("tag") or []) + ["litept", "occ"])),
        "source": {
            "db": database, "clip_collection": raw_clip_name,
            "frame_collection": raw_frame_collection,
            "raw_id": str(raw_clip.get("_id")), "clip_id": clip_id,
        },
        "frame_collection": occ_frame_name, "frame_count": len(prepared),
        "start_timestamp": min(timestamps), "end_timestamp": max(timestamps),
        "model": {"name": "litept-small-waymo-semseg", "version": model_version, "taxonomy": "waymo-22"},
        "static_agg": static, "taxonomy": index.get("taxonomy"), "defaults": index.get("defaults"),
        "geometry_quality": index.get("geometry_quality", index.get("pose_quality")),
        "status": {"state": "complete", "processed_frames": len(prepared), "failed_frames": 0},
        "updated_at": now,
    }
    occ_clips.update_one(
        {"source.clip_collection": raw_clip_name, "source.clip_id": clip_id, "model.version": model_version},
        {"$set": clip_doc, "$setOnInsert": {"created_at": now}}, upsert=True,
    )
    print("content-addressed OCC store complete", flush=True)
    return 0

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", default="")
    ap.add_argument("--occ-frame-collection", default="")
    ap.add_argument("--occ-clip-collection", default="")
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--mongo-uri", default=DEFAULT_URI)
    ap.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    ap.add_argument("--asset-storage", choices=("file", "gridfs"), default="file")
    ap.add_argument("--gridfs-bucket", default="occ_blobs")
    ap.add_argument("--split-tag", choices=("train", "val"), default="")
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--backup-root", default="data/robotruck_clips_backup")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    return run(
        scene=args.scene,
        raw_frame_collection=args.raw_frame_collection,
        raw_clip_collection=args.raw_clip_collection,
        occ_frame_collection=args.occ_frame_collection,
        occ_clip_collection=args.occ_clip_collection,
        database=args.database,
        mongo_uri=args.mongo_uri,
        asset_root=args.asset_root,
        asset_storage=args.asset_storage,
        gridfs_bucket=args.gridfs_bucket,
        split_tag=args.split_tag,
        model_version=args.model_version,
        backup_root=args.backup_root,
        resume=args.resume,
        write=args.write,
    )

if __name__ == "__main__":
    raise SystemExit(main())
