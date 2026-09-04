#!/usr/bin/env python3
"""Round-robin tag batch: pipeline(stride=5) → GSS publish. Production manifest for lidar14_0813."""
from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone

from pymongo import MongoClient

from paths import ensure_import_path

ensure_import_path(repo=True)

import gss_mongo as GSS
import pipeline as PIPELINE
import store as STORE

TAGS = [
    "highway", "mountainous_winding_road", "bridge", "urban_street_scene",
    "interchange_ramp", "toll_station", "tunnel", "logistics_park",
    "highway_service_area", "bus", "engineering_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "construction_sign", "crash_cushion",
    "water_barrier", "flexible_delineator_post", "triangular_warning_sign",
    "night", "rain", "fog", "ego_turning", "lanechange", "stop", "cutin",
    "highway_congestion", "engineering_vehicle_0731", "motorcycle_0731",
    "fog_0731", "rain_0731", "backlight_0731", "crashed_vehicle_0731",
]


def free_gib(db) -> float:
    stats = db.command({"dbStats": 1, "scale": 1})
    return (stats.get("fsTotalSize", 0) - stats.get("fsUsedSize", 0)) / 1024**3


def ordered_clips(collection) -> list[tuple[str, dict]]:
    by_tag = {tag: list(collection.find({"tag": tag}).sort("clip_id", 1)) for tag in TAGS}
    output: list[tuple[str, dict]] = []
    for index in range(max(map(len, by_tag.values()))):
        for tag in TAGS:
            if index < len(by_tag[tag]):
                output.append((tag, by_tag[tag][index]))
    return output


def publish_gss(db, raw_clip: dict, tag: str, version: str) -> None:
    clip_id = raw_clip["clip_id"]
    frames = list(db.occ_data_frames_lidar14_0813.find({
        "source.clip_id": clip_id, "model.version": "v1"
    }).sort("timestamp", 1))
    if not frames:
        raise RuntimeError(f"no OCC frames stored for clip_id={clip_id}")
    document = GSS.build_gss_document(
        tag=tag,
        version=version,
        run_id=f"litept-s5-0813-{clip_id}-v1",
        clips=[GSS.build_gss_clip(raw_clip, frames)],
        producer={
            "name": "LitePT",
            "branch": "dev_occ",
            "model": frames[0].get("model"),
            "stride": 5,
            "raw_lidar_sensor": "lidar_merge_deskew",
        },
        created_by="LitePT batch",
    )
    GSS.publish(db.occ_data_groundtruths_lidar14_0813, document)


def record_failure(db, *, clip_id: str, tag: str, stage: str, error: Exception) -> None:
    db.occ_data_batch_failures_lidar14_0813.update_one(
        {"clip_id": clip_id, "stride": 5, "stage": stage},
        {"$set": {
            "tag": tag,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "updated_at": datetime.now(timezone.utc),
        }, "$inc": {"attempts": 1}},
        upsert=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mongo-uri", default=STORE.DEFAULT_URI)
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-free-gib", type=float, default=15.0)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--dataset-version", default="lidar14-0813-occ-s5-v1")
    args = ap.parse_args(argv)
    if args.stride != 5:
        raise ValueError("this production manifest is fixed to stride=5")

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    clips = ordered_clips(db.raw_data_clips_lidar14_0813)
    completed = set(db.occ_data_clips_lidar14_0813.distinct(
        "source.clip_id", {"model.version": "v1", "status.state": "complete"}
    ))
    done_this_run = 0
    print(f"manifest clips={len(clips)} precompleted={len(completed)} free={free_gib(db):.2f}GiB", flush=True)
    for number, (tag, raw_clip) in enumerate(clips, 1):
        clip_id = raw_clip["clip_id"]
        if clip_id in completed:
            publish_gss(db, raw_clip, tag, args.dataset_version)
            print(f"[{number}/{len(clips)}] published existing tag={tag} clip={clip_id}", flush=True)
            continue
        available = free_gib(db)
        if available < args.min_free_gib:
            print(f"STOP storage guard free={available:.2f}GiB < {args.min_free_gib:.2f}GiB", flush=True)
            return 0
        if args.max_clips and done_this_run >= args.max_clips:
            print(f"STOP max_clips={args.max_clips}", flush=True)
            return 0
        scene_name = f"batch_s5_{clip_id}"
        print(f"[{number}/{len(clips)}] START tag={tag} clip={clip_id} free={available:.2f}GiB", flush=True)
        try:
            rc = PIPELINE.run(
                raw_frame_collection="raw_data_frames_lidar14_0813",
                raw_clip_collection="raw_data_clips_lidar14_0813",
                clip_id=clip_id,
                scene_name=scene_name,
                stride=args.stride,
                write=True,
            )
            if rc:
                raise RuntimeError(f"pipeline rc={rc}")
        except Exception as error:
            record_failure(db, clip_id=clip_id, tag=tag, stage="pipeline", error=error)
            print(f"[{number}/{len(clips)}] FAILED pipeline tag={tag} clip={clip_id}: {error}", flush=True)
            continue
        try:
            publish_gss(db, raw_clip, tag, args.dataset_version)
        except Exception as error:
            record_failure(db, clip_id=clip_id, tag=tag, stage="gss_publish", error=error)
            print(f"[{number}/{len(clips)}] FAILED gss_publish tag={tag} clip={clip_id}: {error}", flush=True)
            continue
        completed.add(clip_id)
        done_this_run += 1
        print(f"[{number}/{len(clips)}] DONE tag={tag} clip={clip_id} free={free_gib(db):.2f}GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
