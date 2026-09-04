#!/usr/bin/env python3
"""Build and publish LitePT occupancy results using the GSS document layout.

The GSS-facing collection embeds ``clips -> frames -> occupancy``.  Large
arrays remain in GridFS and are referenced by the frame occupancy document.
The lower-level ``occ_data_frames_*`` and ``occ_data_clips_*`` collections are
kept as inference/run records and are not the primary GSS dataset interface.
"""
from __future__ import annotations

import argparse
import copy
import os
from datetime import datetime, timezone
from typing import Any, Iterable

from bson import BSON
from pymongo import ASCENDING, MongoClient


DEFAULT_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def collection_suffix(raw_collection: str, kind: str) -> str:
    prefix = f"raw_data_{kind}"
    if not raw_collection.startswith(prefix):
        raise ValueError(f"expected collection prefix {prefix!r}: {raw_collection!r}")
    return raw_collection[len(prefix):].lstrip("_")


def groundtruth_collection_name(raw_frame_collection: str) -> str:
    suffix = collection_suffix(raw_frame_collection, "frames")
    return "occ_data_groundtruths" + (f"_{suffix}" if suffix else "")


def _source_reference(frame: dict[str, Any]) -> dict[str, Any]:
    source = frame.get("source") or {}
    return {
        "database": source.get("db"),
        "frame_collection": source.get("frame_collection"),
        "clip_collection": source.get("clip_collection"),
        "document_id": source.get("raw_id"),
        "md5": source.get("frame_md5") or frame.get("md5"),
        "lidar_merge_md5": source.get("lidar_merge_md5") or frame.get("md5"),
        "timestamp": frame.get("timestamp"),
        "clip_id": source.get("clip_id") or frame.get("clip_id"),
    }


def build_gss_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert a LitePT inference frame document to a GSS OCC frame."""
    assets = frame.get("assets") or {}
    occ_assets = assets.get("occupancy")
    if not isinstance(occ_assets, dict) or not occ_assets:
        raise ValueError("OCC frame has no assets.occupancy")
    grid = copy.deepcopy(frame.get("grid") or {})
    coordinate = copy.deepcopy(frame.get("coordinate") or {})
    if isinstance(coordinate, str):
        coordinate = {"name": coordinate}
    coordinate.setdefault("name", coordinate.pop("frame", "vehicle"))
    coordinate.setdefault("convention", "x_lateral_y_forward_z_up")
    stats = frame.get("stats") or {}
    return {
        "md5": frame.get("md5"),
        "timestamp": frame.get("timestamp"),
        "raw_data": _source_reference(frame),
        "occupancy": {
            "schema_version": "litept_sparse_occ/v1",
            "format": "sparse_voxel",
            "coordinate_system": coordinate,
            "grid": grid,
            "voxel_count": stats.get("n_occ"),
            "assets": copy.deepcopy(occ_assets),
            "statistics": copy.deepcopy(stats),
        },
    }


def build_gss_clip(
    raw_clip: dict[str, Any], frames: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    gss_frames = [build_gss_frame(frame) for frame in frames]
    gss_frames.sort(key=lambda item: item.get("timestamp") or 0)
    timestamps = [f["timestamp"] for f in gss_frames if f.get("timestamp") is not None]
    return {
        "clip_id": raw_clip.get("clip_id"),
        "md5": raw_clip.get("md5"),
        "timestamp": raw_clip.get("timestamp"),
        "bag_md5": raw_clip.get("bag_md5"),
        "bag_name": raw_clip.get("bag_name"),
        "bag_path": raw_clip.get("bag_path"),
        "clip_name": raw_clip.get("clip_name"),
        "meta": copy.deepcopy(raw_clip.get("meta")),
        "sensor_list": copy.deepcopy(raw_clip.get("sensor_list") or []),
        "start_timestamp": raw_clip.get("start_timestamp") or (min(timestamps) if timestamps else None),
        "end_timestamp": raw_clip.get("end_timestamp") or (max(timestamps) if timestamps else None),
        "key_timestamp": raw_clip.get("key_timestamp"),
        "frames": gss_frames,
        "frame_count": len(gss_frames),
    }


def build_gss_document(
    *,
    tag: str,
    version: str,
    run_id: str,
    clips: Iterable[dict[str, Any]],
    producer: dict[str, Any],
    status: str = "complete",
    created_by: str = "LitePT",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    if not tag or not version or not run_id:
        raise ValueError("tag, version and run_id are required")
    now = timestamp or utc_now()
    clip_list = list(clips)
    document = {
        "schema_version": "gss_occ_groundtruth/v1",
        "tag": tag,
        "version": version,
        "run_id": run_id,
        "status": status,
        "producer": copy.deepcopy(producer),
        "clips": clip_list,
        "clip_count": len(clip_list),
        "frame_count": sum(int(clip.get("frame_count") or 0) for clip in clip_list),
        "created_by": created_by,
        "created_timestamp": now,
        "update_by": created_by,
        "updated_timestamp": now,
    }
    size = len(BSON.encode(document))
    if size >= MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"GSS OCC document is {size} bytes and exceeds MongoDB's 16 MiB limit; "
            "split the run into multiple tag/version documents"
        )
    return document


def create_indexes(collection) -> None:
    collection.create_index([("run_id", ASCENDING)], unique=True, name="run_id_unique")
    collection.create_index(
        [("tag", ASCENDING), ("version", ASCENDING), ("status", ASCENDING)],
        name="tag_version_status",
    )
    # Keep these as separate multikey indexes, matching existing OD collections.
    collection.create_index([("clips.clip_id", ASCENDING)], name="clips_clip_id")
    collection.create_index([("clips.md5", ASCENDING)], name="clips_md5")
    collection.create_index([("clips.frames.md5", ASCENDING)], name="frames_md5")
    collection.create_index([("clips.frames.timestamp", ASCENDING)], name="frames_timestamp")
    collection.create_index(
        [("clips.frames.raw_data.frame_collection", ASCENDING)], name="raw_frame_collection"
    )


def publish(collection, document: dict[str, Any]) -> Any:
    """Idempotently publish one immutable run identity to the GSS collection."""
    create_indexes(collection)
    existing = collection.find_one({"run_id": document["run_id"]}, {"_id": 1})
    if existing:
        collection.replace_one({"_id": existing["_id"]}, document)
        return existing["_id"]
    return collection.insert_one(document).inserted_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", required=True)
    ap.add_argument("--occ-frame-collection", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--mongo-uri", default=DEFAULT_URI)
    ap.add_argument("--created-by", default="LitePT")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    occ_frames = list(db[args.occ_frame_collection].find({"run_id": args.run_id}))
    if not occ_frames:
        raise RuntimeError(f"no OCC frames found for run_id={args.run_id}")
    clips = []
    for clip_id in sorted({frame.get("clip_id") for frame in occ_frames}, key=str):
        raw_clip = db[args.raw_clip_collection].find_one({"clip_id": clip_id})
        if raw_clip is None:
            raise RuntimeError(f"raw clip_id={clip_id!r} not found")
        clips.append(build_gss_clip(raw_clip, (f for f in occ_frames if f.get("clip_id") == clip_id)))
    first = occ_frames[0]
    model = first.get("model") or {}
    document = build_gss_document(
        tag=args.tag,
        version=args.version,
        run_id=args.run_id,
        clips=clips,
        producer={"name": "LitePT", "model": model},
        created_by=args.created_by,
    )
    target_name = groundtruth_collection_name(args.raw_frame_collection)
    print({
        "mode": "write" if args.write else "dry-run",
        "target": f"{args.database}:{target_name}",
        "query": {"tag": args.tag, "version": args.version, "status": "complete"},
        "run_id": args.run_id,
        "clip_count": document["clip_count"],
        "frame_count": document["frame_count"],
        "bson_bytes": len(BSON.encode(document)),
    })
    if args.write:
        inserted_id = publish(db[target_name], document)
        print({"_id": str(inserted_id)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
