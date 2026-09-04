#!/usr/bin/env python3
"""Ingest an exported Robotruck Occ scene package into MongoDB.

The command treats the existing ``raw_data_*`` collections as immutable source
contracts.  Every Occ frame must resolve to one raw frame by lidar/frame MD5
before it can be written.  Large binary assets stay outside MongoDB; documents
contain validated URIs plus dtype/shape metadata.

Dry-run (default)::

    python tools/occ/_impl/ingest.py \
      --scene exp/robotruck/occ_scenes/stop_... \
      --raw-frame-collection raw_data_frames_braking_matrix

Write idempotently after reviewing the dry-run summary::

    python tools/occ/_impl/ingest.py ... --write
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, MongoClient


DEFAULT_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)
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
    return "occ_data_" + raw_name[len("raw_data_") :]


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


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


def resolve_asset_uri(
    uri: str,
    *,
    scene_root: Path,
    asset_uri_root: str | None,
) -> str:
    if "://" in uri or uri.startswith("/"):
        return uri
    rel = Path(uri)
    local = (scene_root / rel).resolve()
    if not local.is_file():
        raise FileNotFoundError(f"scene asset does not exist: {local}")
    if asset_uri_root:
        return asset_uri_root.rstrip("/") + "/" + rel.as_posix()
    return str(local)


def rewrite_asset_tree(
    value: Any,
    *,
    scene_root: Path,
    asset_uri_root: str | None,
) -> Any:
    if isinstance(value, list):
        return [
            rewrite_asset_tree(v, scene_root=scene_root, asset_uri_root=asset_uri_root)
            for v in value
        ]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key == "uri" and isinstance(item, str):
            out[key] = resolve_asset_uri(
                item, scene_root=scene_root, asset_uri_root=asset_uri_root
            )
        else:
            out[key] = rewrite_asset_tree(
                item, scene_root=scene_root, asset_uri_root=asset_uri_root
            )
    return out


def raw_lidar_md5(raw: dict[str, Any]) -> str | None:
    sensors = ((raw.get("dependency") or {}).get("sensors") or {})
    # The 0813 production collection contains both the original merged cloud
    # and an ego/object deskewed derivative.  Prefer deskew for OCC, while
    # retaining compatibility with older collections and names.
    for name in ("lidar_merge_deskew", "lidar_merge", "lidar_merge_nodeskew"):
        value = ((sensors.get(name) or {}).get("md5"))
        if isinstance(value, str) and MD5_RE.fullmatch(value):
            return value.lower()
    return None


def raw_camera_refs(raw: dict[str, Any], rawdata_root: str) -> list[dict[str, Any]]:
    sensors = ((raw.get("dependency") or {}).get("sensors") or {})
    refs: list[dict[str, Any]] = []
    for name, sensor in sorted(sensors.items()):
        if not name.startswith("camera") or not isinstance(sensor, dict):
            continue
        md5 = sensor.get("md5")
        if not isinstance(md5, str) or not MD5_RE.fullmatch(md5):
            continue
        md5 = md5.lower()
        refs.append(
            {
                "name": name,
                "md5": md5,
                "timestamp": sensor.get("timestamp"),
                "uri": (
                    f"{rawdata_root.rstrip('/')}/camera/{md5[:2]}/{md5[2:4]}/{md5[4:]}.jpg"
                ),
            }
        )
    return refs


def scene_fingerprint(index: dict[str, Any]) -> str:
    payload = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_frame_document(
    *,
    scene_root: Path,
    scene_index: dict[str, Any],
    frame_entry: dict[str, Any],
    frame_meta: dict[str, Any],
    raw_frame: dict[str, Any],
    database: str,
    raw_frame_collection: str,
    raw_clip_collection: str,
    asset_uri_root: str | None,
    rawdata_root: str,
    model_name: str,
    model_version: str,
    checkpoint: str | None,
    ingested_at: datetime,
) -> dict[str, Any]:
    md5 = raw_lidar_md5(raw_frame) or raw_frame.get("md5")
    if not isinstance(md5, str) or not MD5_RE.fullmatch(md5):
        raise ValueError("raw frame has no valid frame/lidar MD5")
    md5 = md5.lower()
    timestamp = raw_frame.get("timestamp", frame_entry.get("timestamp"))
    assets = rewrite_asset_tree(
        frame_meta.get("assets") or {},
        scene_root=scene_root,
        asset_uri_root=asset_uri_root,
    )
    raw_id = raw_frame.get("_id")
    doc: dict[str, Any] = {
        "schema_version": "litept_occ_frame/v1",
        "md5": md5,
        "timestamp": timestamp,
        "clip_id": raw_frame.get("clip_id"),
        "bag_name": raw_frame.get("bag_name"),
        "tag": sorted(set((raw_frame.get("tag") or []) + ["litept", "occ", "semseg"])),
        "source": {
            "db": database,
            "clip_collection": raw_clip_collection,
            "frame_collection": raw_frame_collection,
            "raw_id": str(raw_id) if raw_id is not None else None,
            "frame_md5": raw_frame.get("md5"),
            "lidar_merge_md5": raw_lidar_md5(raw_frame),
            "scene_id": scene_index.get("scene_id"),
            "scene_frame_id": frame_entry.get("frame_id"),
        },
        "model": {
            "name": model_name,
            "version": model_version,
            "checkpoint": checkpoint,
            "grid_sample_size": 0.05,
            "taxonomy": "waymo-22",
        },
        "coordinate": frame_meta.get("coordinate"),
        "grid": frame_meta.get("grid"),
        "stats": frame_meta.get("stats") or {
            "n_occ": frame_entry.get("n_occ"),
            "n_points": frame_entry.get("n_points"),
        },
        "assets": assets,
        "raw_assets": {
            "lidar_merge": {
                "md5": raw_lidar_md5(raw_frame),
                "uri": (
                    f"{rawdata_root.rstrip('/')}/lidar/{md5[:2]}/{md5[2:4]}/{md5[4:]}.bin"
                ),
            },
            "cameras": raw_camera_refs(raw_frame, rawdata_root),
        },
        "ego_pose": ((raw_frame.get("dependency") or {}).get("ego_pose")),
        "provenance": {
            "scene_schema_version": scene_index.get("schema_version"),
            "scene_created_at": scene_index.get("created_at"),
            "ingested_at": ingested_at,
        },
        "updated_at": ingested_at,
    }
    return doc


def build_clip_document(
    *,
    scene_root: Path,
    scene_index: dict[str, Any],
    raw_clip: dict[str, Any] | None,
    raw_clip_id: Any,
    database: str,
    raw_frame_collection: str,
    raw_clip_collection: str,
    occ_frame_collection: str,
    asset_uri_root: str | None,
    model_name: str,
    model_version: str,
    checkpoint: str | None,
    processed_frames: list[dict[str, Any]],
    ingested_at: datetime,
) -> dict[str, Any]:
    raw_clip = raw_clip or {}
    timestamps = [d.get("timestamp") for d in processed_frames if d.get("timestamp") is not None]
    static_agg = scene_index.get("static_agg")
    if static_agg:
        static_agg = rewrite_asset_tree(
            static_agg,
            scene_root=scene_root,
            asset_uri_root=asset_uri_root,
        )
    raw_id = raw_clip.get("_id")
    return {
        "schema_version": "litept_occ_clip/v1",
        "clip_id": raw_clip_id,
        "clip_name": raw_clip.get("clip_name"),
        "bag_name": raw_clip.get("bag_name") or (processed_frames[0].get("bag_name") if processed_frames else None),
        "bag_path": raw_clip.get("bag_path"),
        "tag": sorted(set((raw_clip.get("tag") or []) + ["litept", "occ"])),
        "source": {
            "db": database,
            "clip_collection": raw_clip_collection,
            "frame_collection": raw_frame_collection,
            "raw_id": str(raw_id) if raw_id is not None else None,
            "clip_id": raw_clip_id,
            "scene_id": scene_index.get("scene_id"),
            "scene_fingerprint": scene_fingerprint(scene_index),
        },
        "model": {
            "name": model_name,
            "version": model_version,
            "checkpoint": checkpoint,
            "taxonomy": "waymo-22",
            "occ_voxel": ((scene_index.get("defaults") or {}).get("occ_voxel")),
            "static_voxel": (static_agg or {}).get("voxel"),
        },
        "frame_collection": occ_frame_collection,
        "frame_count": len(processed_frames),
        "start_timestamp": min(timestamps) if timestamps else None,
        "end_timestamp": max(timestamps) if timestamps else None,
        "static_agg": static_agg,
        "taxonomy": scene_index.get("taxonomy"),
        "defaults": scene_index.get("defaults"),
        "status": {
            "state": "complete",
            "processed_frames": len(processed_frames),
            "failed_frames": 0,
        },
        "provenance": {
            "scene_schema_version": scene_index.get("schema_version"),
            "scene_created_at": scene_index.get("created_at"),
            "ingested_at": ingested_at,
        },
        "updated_at": ingested_at,
    }


def find_raw_frame(
    collection, md5: str, timestamp: int | str | None = None
) -> dict[str, Any] | None:
    # Timestamp is indexed in the production frame collections and is the
    # preferred lookup for derived deskew assets whose MD5 path is not indexed.
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
    return collection.find_one(
        {
            "$or": [
                {"md5": md5},
                {"dependency.sensors.lidar_merge.md5": md5},
                {"dependency.sensors.lidar_merge_deskew.md5": md5},
                {"dependency.sensors.lidar_merge_nodeskew.md5": md5},
            ]
        }
    )


def create_indexes(frame_collection, clip_collection) -> None:
    frame_collection.create_index(
        [
            ("source.frame_collection", ASCENDING),
            ("source.frame_md5", ASCENDING),
            ("model.version", ASCENDING),
        ],
        unique=True,
        name="source_frame_model_unique",
    )
    frame_collection.create_index(
        [("bag_name", ASCENDING), ("timestamp", ASCENDING)],
        name="bag_timestamp",
    )
    frame_collection.create_index([("md5", ASCENDING)], name="frame_md5")
    clip_collection.create_index(
        [
            ("source.clip_collection", ASCENDING),
            ("source.clip_id", ASCENDING),
            ("model.version", ASCENDING),
        ],
        unique=True,
        name="source_clip_model_unique",
    )
    clip_collection.create_index([("bag_name", ASCENDING)], name="bag_name")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="Exported scene directory containing index.json")
    ap.add_argument("--backup-root", default="data/robotruck_clips_backup")
    ap.add_argument("--mongo-uri", default=DEFAULT_URI)
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", default="", help="Default: replace raw_data_frames with raw_data_clips")
    ap.add_argument("--occ-frame-collection", default="", help="Default: raw_data_ → occ_data_")
    ap.add_argument("--occ-clip-collection", default="", help="Default: raw_data_ → occ_data_")
    ap.add_argument("--asset-uri-root", default="", help="Rewrite scene-relative assets under this URI root")
    ap.add_argument("--rawdata-root", default="/data/rawdata")
    ap.add_argument("--model-name", default="litept-small-waymo-semseg")
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--checkpoint", default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N frames")
    ap.add_argument("--write", action="store_true", help="Create indexes and upsert; default is dry-run")
    args = ap.parse_args()
    if args.write:
        raise RuntimeError(
            "legacy URI writes are disabled; use store_robotruck_occ_gridfs.py "
            "or run_robotruck_occ_mongo_pipeline.py"
        )

    scene_root = Path(args.scene).resolve()
    index_path = scene_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    scene_index = load_json(index_path)
    raw_clip_collection = args.raw_clip_collection or infer_collection(
        args.raw_frame_collection, "frames", "clips"
    )
    occ_frame_collection = args.occ_frame_collection or infer_occ_collection(
        args.raw_frame_collection
    )
    occ_clip_collection = args.occ_clip_collection or infer_occ_collection(
        raw_clip_collection
    )
    backup_root = Path(args.backup_root).resolve() / scene_root.name
    backup_dir = backup_root if backup_root.is_dir() else None

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    raw_frames = db[args.raw_frame_collection]
    raw_clips = db[raw_clip_collection]
    ingested_at = utc_now()
    frame_entries = list(scene_index.get("frames") or [])
    if args.limit > 0:
        frame_entries = frame_entries[: args.limit]

    frame_docs: list[dict[str, Any]] = []
    raw_clip_ids: set[Any] = set()
    missing: list[str] = []
    for entry in frame_entries:
        md5 = md5_from_backup(entry, backup_dir)
        ts = str(entry.get("timestamp") or entry.get("frame_id") or "")
        if md5 is None:
            missing.append(f"timestamp={ts}: no raw MD5 in scene/backup")
            continue
        raw = find_raw_frame(raw_frames, md5)
        if raw is None:
            missing.append(f"timestamp={ts} md5={md5}: absent from {args.raw_frame_collection}")
            continue
        if raw.get("clip_id") is not None:
            raw_clip_ids.add(raw["clip_id"])
        meta_uri = entry.get("meta_uri") or f"frames/{ts}/meta.json"
        frame_meta = load_json(scene_root / meta_uri)
        frame_docs.append(
            build_frame_document(
                scene_root=scene_root,
                scene_index=scene_index,
                frame_entry=entry,
                frame_meta=frame_meta,
                raw_frame=raw,
                database=args.database,
                raw_frame_collection=args.raw_frame_collection,
                raw_clip_collection=raw_clip_collection,
                asset_uri_root=args.asset_uri_root or None,
                rawdata_root=args.rawdata_root,
                model_name=args.model_name,
                model_version=args.model_version,
                checkpoint=args.checkpoint or None,
                ingested_at=ingested_at,
            )
        )

    if missing:
        preview = "\n".join("  - " + item for item in missing[:20])
        more = f"\n  ... {len(missing) - 20} more" if len(missing) > 20 else ""
        raise RuntimeError(f"raw traceability failed for {len(missing)} frame(s):\n{preview}{more}")
    if not frame_docs:
        raise RuntimeError("no frames resolved; nothing to ingest")
    if len(raw_clip_ids) != 1:
        raise RuntimeError(
            f"scene must resolve to exactly one raw clip_id; got {sorted(map(str, raw_clip_ids))}"
        )
    raw_clip_id = next(iter(raw_clip_ids))
    raw_clip = raw_clips.find_one({"clip_id": raw_clip_id})
    clip_doc = build_clip_document(
        scene_root=scene_root,
        scene_index=scene_index,
        raw_clip=raw_clip,
        raw_clip_id=raw_clip_id,
        database=args.database,
        raw_frame_collection=args.raw_frame_collection,
        raw_clip_collection=raw_clip_collection,
        occ_frame_collection=occ_frame_collection,
        asset_uri_root=args.asset_uri_root or None,
        model_name=args.model_name,
        model_version=args.model_version,
        checkpoint=args.checkpoint or None,
        processed_frames=frame_docs,
        ingested_at=ingested_at,
    )

    summary = {
        "mode": "write" if args.write else "dry-run",
        "scene": str(scene_root),
        "database": args.database,
        "raw": {
            "clips": raw_clip_collection,
            "frames": args.raw_frame_collection,
            "clip_id": raw_clip_id,
        },
        "occ": {
            "clips": occ_clip_collection,
            "frames": occ_frame_collection,
        },
        "frames": len(frame_docs),
        "first_md5": frame_docs[0]["md5"],
        "last_md5": frame_docs[-1]["md5"],
        "asset_uri_root": args.asset_uri_root or str(scene_root),
        "model_version": args.model_version,
    }
    print(json.dumps(summary, default=str, ensure_ascii=False, indent=2))
    if not args.write:
        print("dry-run complete; pass --write to create indexes and upsert")
        return 0

    occ_frames = db[occ_frame_collection]
    occ_clips = db[occ_clip_collection]
    create_indexes(occ_frames, occ_clips)
    inserted = matched = modified = upserted = 0
    for doc in frame_docs:
        result = occ_frames.update_one(
            {
                "source.frame_collection": args.raw_frame_collection,
                "source.frame_md5": doc["source"]["frame_md5"],
                "model.version": args.model_version,
            },
            {"$set": doc, "$setOnInsert": {"created_at": ingested_at}},
            upsert=True,
        )
        matched += result.matched_count
        modified += result.modified_count
        if result.upserted_id is not None:
            inserted += 1
            upserted += 1
    clip_result = occ_clips.update_one(
        {
            "source.clip_collection": raw_clip_collection,
            "source.clip_id": raw_clip_id,
            "model.version": args.model_version,
        },
        {"$set": clip_doc, "$setOnInsert": {"created_at": ingested_at}},
        upsert=True,
    )
    print(
        json.dumps(
            {
                "write_complete": True,
                "frame_inserted": inserted,
                "frame_matched": matched,
                "frame_modified": modified,
                "clip_upserted": clip_result.upserted_id is not None,
                "clip_matched": clip_result.matched_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
