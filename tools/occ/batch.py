#!/usr/bin/env python3
"""Prod: round-robin tag batch → pipeline → GSS publish (collections not hardcoded)."""
from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone

from pymongo import MongoClient

from paths import DEFAULT_URI, ensure_import_path

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
    n = max((len(v) for v in by_tag.values()), default=0)
    out: list[tuple[str, dict]] = []
    for i in range(n):
        for tag in TAGS:
            if i < len(by_tag[tag]):
                out.append((tag, by_tag[tag][i]))
    return out


def publish_gss(db, *, raw_clip, tag, version, stride, occ_frames_col, gt_col) -> None:
    clip_id = raw_clip["clip_id"]
    frames = list(db[occ_frames_col].find({
        "source.clip_id": clip_id, "model.version": "v1"
    }).sort("timestamp", 1))
    if not frames:
        raise RuntimeError(f"no OCC frames in {occ_frames_col} for clip_id={clip_id}")
    doc = GSS.build_gss_document(
        tag=tag, version=version, run_id=f"litept-s{stride}-{clip_id}-v1",
        clips=[GSS.build_gss_clip(raw_clip, frames)],
        producer={
            "name": "LitePT", "branch": "dev_occ", "model": frames[0].get("model"),
            "stride": stride, "raw_lidar_sensor": "lidar_merge_deskew",
        },
        created_by="LitePT batch",
    )
    GSS.publish(db[gt_col], doc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mongo-uri", default=DEFAULT_URI)
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", default="")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-free-gib", type=float, default=15.0)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--dataset-version", default="occ-s5-v1")
    ap.add_argument("--scene-prefix", default="batch_s5_")
    args = ap.parse_args(argv)

    raw_frames = args.raw_frame_collection
    raw_clips = args.raw_clip_collection or STORE.infer_collection(raw_frames, "frames", "clips")
    occ_frames = STORE.infer_occ_collection(raw_frames)
    occ_clips = STORE.infer_occ_collection(raw_clips)
    gt_col = GSS.groundtruth_collection_name(raw_frames)
    fail_col = occ_clips.replace("occ_data_clips", "occ_data_batch_failures", 1)

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    clips = ordered_clips(db[raw_clips])
    completed = set(db[occ_clips].distinct(
        "source.clip_id", {"model.version": "v1", "status.state": "complete"}
    ))
    done = 0
    print(f"clips={len(clips)} done={len(completed)} free={free_gib(db):.2f}GiB "
          f"raw={raw_clips} occ={occ_clips} gt={gt_col}", flush=True)

    for n, (tag, raw_clip) in enumerate(clips, 1):
        clip_id = raw_clip["clip_id"]
        if clip_id in completed:
            publish_gss(db, raw_clip=raw_clip, tag=tag, version=args.dataset_version,
                        stride=args.stride, occ_frames_col=occ_frames, gt_col=gt_col)
            print(f"[{n}/{len(clips)}] published {tag} {clip_id}", flush=True)
            continue
        if free_gib(db) < args.min_free_gib:
            print("STOP storage guard", flush=True)
            return 0
        if args.max_clips and done >= args.max_clips:
            print(f"STOP max_clips={args.max_clips}", flush=True)
            return 0
        scene = f"{args.scene_prefix}{clip_id}"
        print(f"[{n}/{len(clips)}] START {tag} {clip_id}", flush=True)
        try:
            rc = PIPELINE.run(
                raw_frame_collection=raw_frames, raw_clip_collection=raw_clips,
                clip_id=clip_id, scene_name=scene, stride=args.stride, write=True,
            )
            if rc:
                raise RuntimeError(f"pipeline rc={rc}")
            publish_gss(db, raw_clip=raw_clip, tag=tag, version=args.dataset_version,
                        stride=args.stride, occ_frames_col=occ_frames, gt_col=gt_col)
        except Exception as err:
            db[fail_col].update_one(
                {"clip_id": clip_id, "stride": args.stride, "stage": "pipeline"},
                {"$set": {
                    "tag": tag, "error_type": type(err).__name__, "error": str(err),
                    "traceback": traceback.format_exc(),
                    "updated_at": datetime.now(timezone.utc),
                }, "$inc": {"attempts": 1}},
                upsert=True,
            )
            print(f"[{n}/{len(clips)}] FAILED {clip_id}: {err}", flush=True)
            continue
        completed.add(clip_id)
        done += 1
        print(f"[{n}/{len(clips)}] DONE {clip_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
