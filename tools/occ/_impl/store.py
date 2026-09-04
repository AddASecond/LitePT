#!/usr/bin/env python3
"""Store OCC metadata in MongoDB and arrays as content-addressed files.

Raw blobs are accepted only from /data/rawdata and /data/rawdata-1..-4.
The scene directory is an inference cache, never a raw-data source.  Large OCC
arrays follow the raw sensor storage pattern; MongoDB stores compact references.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "occ_ingest", ROOT / "tools/occ/_impl/ingest.py"
)
INGEST = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(INGEST)
RAW_ROOTS = tuple(Path(f"/data/rawdata{s}") for s in ("", "-1", "-2", "-3", "-4"))
DEFAULT_ASSET_ROOT = Path("/data/rawdata-4/occupancy")


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


def upload_group(scene: Path, group: dict | None, asset_root: Path):
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
        output[name].update(store(path, asset_root))
    return output


def raw_cameras(raw: dict) -> list[dict]:
    sensors = ((raw.get("dependency") or {}).get("sensors") or {})
    output = []
    for name, sensor in sorted(sensors.items()):
        if not name.startswith("camera") or not isinstance(sensor, dict):
            continue
        md5 = sensor.get("md5")
        if isinstance(md5, str) and INGEST.MD5_RE.fullmatch(md5):
            output.append({"name": name, "md5": md5, "timestamp": sensor.get("timestamp"), "uri": str(resolve_raw(md5, "camera"))})
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", default="")
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--mongo-uri", default=INGEST.DEFAULT_URI)
    ap.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--backup-root", default="data/robotruck_clips_backup")
    ap.add_argument("--resume", action="store_true", help="Skip frames already stored as v2")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    index = INGEST.load_json(scene / "index.json")
    raw_clip_name = args.raw_clip_collection or INGEST.infer_collection(args.raw_frame_collection, "frames", "clips")
    occ_frame_name = INGEST.infer_occ_collection(args.raw_frame_collection)
    occ_clip_name = INGEST.infer_occ_collection(raw_clip_name)
    backup = (ROOT / args.backup_root / scene.name).resolve()
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    raw_frames, raw_clips = db[args.raw_frame_collection], db[raw_clip_name]
    occ_frames, occ_clips = db[occ_frame_name], db[occ_clip_name]

    prepared, clip_ids = [], set()
    for entry in index.get("frames") or []:
        md5 = INGEST.md5_from_backup(entry, backup)
        raw = INGEST.find_raw_frame(raw_frames, md5, entry.get("timestamp")) if md5 else None
        if raw is None:
            raise RuntimeError(f"raw trace failed for scene frame {entry.get('timestamp')}, md5={md5}")
        lidar_md5 = INGEST.raw_lidar_md5(raw) or raw.get("md5")
        lidar_path = resolve_raw(lidar_md5, "lidar")
        cameras = raw_cameras(raw)
        clip_ids.add(raw.get("clip_id"))
        ts = str(entry.get("timestamp") or entry.get("frame_id"))
        meta = INGEST.load_json(scene / (entry.get("meta_uri") or f"frames/{ts}/meta.json"))
        prepared.append((entry, raw, meta, lidar_path, cameras))
    if len(clip_ids) != 1:
        raise RuntimeError(f"expected one raw clip, got {clip_ids}")
    clip_id = next(iter(clip_ids))
    raw_clip = raw_clips.find_one({"clip_id": clip_id})
    if raw_clip is None:
        raise RuntimeError(f"raw clip {clip_id} absent from {raw_clip_name}")

    print(json.dumps({
        "mode": "write-content-addressed" if args.write else "dry-run", "scene": str(scene),
        "frames": len(prepared), "raw_roots": [str(p) for p in RAW_ROOTS],
        "raw": {"frames": args.raw_frame_collection, "clips": raw_clip_name, "clip_id": clip_id},
        "occ": {"frames": occ_frame_name, "clips": occ_clip_name,
                "asset_root": str(args.asset_root.resolve())},
    }, default=str, indent=2), flush=True)
    if not args.write:
        return 0

    INGEST.create_indexes(occ_frames, occ_clips)
    now = INGEST.utc_now()
    for number, (entry, raw, meta, lidar_path, cameras) in enumerate(prepared, 1):
        md5 = INGEST.raw_lidar_md5(raw) or raw["md5"]
        identity = {
            "source.frame_collection": args.raw_frame_collection,
            "source.frame_md5": md5,
            "model.version": args.model_version,
        }
        if args.resume and occ_frames.find_one(
            {**identity, "schema_version": "litept_occ_frame/v2"}, {"_id": 1}
        ):
            continue
        source = {"db": args.database, "frame_collection": args.raw_frame_collection, "clip_collection": raw_clip_name, "raw_id": str(raw.get("_id")), "frame_md5": md5, "clip_id": clip_id}
        current = meta.get("assets") or {}
        assets = {
            "occupancy": upload_group(scene, current.get("occupancy"), args.asset_root),
            "points": upload_group(scene, current.get("points"), args.asset_root),
            "cameras": cameras,
        }
        doc = {
            "schema_version": "litept_occ_frame/v2", "md5": md5,
            "timestamp": raw.get("timestamp", entry.get("timestamp")), "clip_id": clip_id,
            "bag_name": raw.get("bag_name"), "tag": sorted(set((raw.get("tag") or []) + ["litept", "occ", "semseg"])),
            "source": source, "raw_assets": {"lidar_merge": {"md5": md5, "uri": str(lidar_path)}, "cameras": cameras},
            "model": {"name": "litept-small-waymo-semseg", "version": args.model_version, "taxonomy": "waymo-22"},
            "coordinate": meta.get("coordinate"), "grid": meta.get("grid"), "stats": meta.get("stats"),
            "assets": assets, "ego_pose": ((raw.get("dependency") or {}).get("ego_pose")),
            "provenance": {"intermediate_scene": str(scene), "scene_schema_version": index.get("schema_version"), "geometry_quality": index.get("geometry_quality", index.get("pose_quality"))},
            "updated_at": now,
        }
        occ_frames.update_one(
            identity,
            {"$set": doc, "$setOnInsert": {"created_at": now}}, upsert=True,
        )
        if number % 10 == 0 or number == len(prepared):
            print(f"uploaded {number}/{len(prepared)}", flush=True)

    static = upload_group(scene, index.get("static_agg"), args.asset_root)
    timestamps = [raw.get("timestamp") for _, raw, _, _, _ in prepared]
    clip_doc = {
        "schema_version": "litept_occ_clip/v2", "clip_id": clip_id,
        "clip_name": raw_clip.get("clip_name"), "bag_name": raw_clip.get("bag_name"), "bag_path": raw_clip.get("bag_path"),
        "tag": sorted(set((raw_clip.get("tag") or []) + ["litept", "occ"])),
        "source": {"db": args.database, "clip_collection": raw_clip_name, "frame_collection": args.raw_frame_collection, "raw_id": str(raw_clip.get("_id")), "clip_id": clip_id},
        "frame_collection": occ_frame_name, "frame_count": len(prepared), "start_timestamp": min(timestamps), "end_timestamp": max(timestamps),
        "model": {"name": "litept-small-waymo-semseg", "version": args.model_version, "taxonomy": "waymo-22"},
        "static_agg": static, "taxonomy": index.get("taxonomy"), "defaults": index.get("defaults"),
        "geometry_quality": index.get("geometry_quality", index.get("pose_quality")),
        "status": {"state": "complete", "processed_frames": len(prepared), "failed_frames": 0}, "updated_at": now,
    }
    occ_clips.update_one(
        {"source.clip_collection": raw_clip_name, "source.clip_id": clip_id, "model.version": args.model_version},
        {"$set": clip_doc, "$setOnInsert": {"created_at": now}}, upsert=True,
    )
    print("content-addressed OCC ingest complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
