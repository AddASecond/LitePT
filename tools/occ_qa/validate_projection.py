#!/usr/bin/env python3
"""Validate one raw Robotruck LiDAR-to-camera projection independently.

This script intentionally does not import the OCC exporter/viewer or LitePT.  It
reads one raw Mongo frame, the content-addressed raw JPEG and
``lidar_merge_nodeskew`` PCD, interpolates raw ego poses, and projects every raw
point into the requested camera.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pymongo import MongoClient


DEFAULT_MONGO_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)
DEFAULT_ROOTS = [Path(f"/data/rawdata{s}") for s in ("", "-1", "-2", "-3", "-4")]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_path(md5: str, kind: str, suffix: str, roots: list[Path]) -> Path:
    rel = Path(kind) / md5[:2] / md5[2:4] / f"{md5[4:]}{suffix}"
    hits = [root / rel for root in roots if (root / rel).is_file()]
    if not hits:
        raise FileNotFoundError(f"content {md5} not found as {rel} under {roots}")
    return hits[0]


def read_binary_pcd(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: incomplete PCD header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text.startswith("DATA "):
                if text != "DATA binary":
                    raise ValueError(f"{path}: only binary PCD is supported")
                break
        body = f.read()

    header = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key] = values
    names = header["FIELDS"]
    sizes = [int(x) for x in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(x) for x in header.get("COUNT", ["1"] * len(names))]
    if any(c != 1 for c in counts):
        raise ValueError(f"{path}: COUNT != 1 is unsupported")
    dtype_fields = []
    for name, size, typ in zip(names, sizes, types):
        table = {
            ("F", 4): "<f4", ("F", 8): "<f8",
            ("I", 4): "<i4", ("I", 8): "<i8",
            ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
        }
        if (typ, size) not in table:
            raise ValueError(f"{path}: unsupported PCD field {name} {typ}{size}")
        dtype_fields.append((name, table[(typ, size)]))
    arr = np.frombuffer(body, dtype=np.dtype(dtype_fields), count=int(header["POINTS"][0]))
    return {name: np.asarray(arr[name]) for name in names}


def quat_to_rotation(q: dict) -> np.ndarray:
    x, y, z, w = (float(q[k]) for k in ("x", "y", "z", "w"))
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y*y + z*z), s * (x*y - z*w), s * (x*z + y*w)],
        [s * (x*y + z*w), 1 - s * (x*x + z*z), s * (y*z - x*w)],
        [s * (x*z - y*w), s * (y*z + x*w), 1 - s * (x*x + y*y)],
    ], dtype=np.float64)


def pose_matrix(pose: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotation(pose["orientation"])
    T[:3, 3] = [float(pose["position"][k]) for k in ("x", "y", "z")]
    return T


def pose_stamp_ns(ego: dict, fallback: int) -> int:
    stamp = (ego.get("header") or {}).get("stamp") or {}
    if "sec" in stamp:
        return int(stamp["sec"]) * 1_000_000_000 + int(stamp.get("nanosec", 0))
    return int(fallback)


def interpolate_pose(samples: list[tuple[int, dict]], timestamp_ns: int) -> np.ndarray:
    times = [x[0] for x in samples]
    hi = bisect.bisect_left(times, int(timestamp_ns))
    if hi <= 0:
        return pose_matrix(samples[0][1])
    if hi >= len(samples):
        return pose_matrix(samples[-1][1])
    ta, pa = samples[hi - 1]
    tb, pb = samples[hi]
    alpha = float(timestamp_ns - ta) / float(tb - ta)
    qa = np.array([pa["orientation"][k] for k in ("x", "y", "z", "w")], np.float64)
    qb = np.array([pb["orientation"][k] for k in ("x", "y", "z", "w")], np.float64)
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = (1.0 - alpha) * qa + alpha * qb
    q /= np.linalg.norm(q)
    pose = {
        "position": {
            k: (1.0 - alpha) * float(pa["position"][k]) + alpha * float(pb["position"][k])
            for k in ("x", "y", "z")
        },
        "orientation": dict(zip(("x", "y", "z", "w"), q.tolist())),
    }
    return pose_matrix(pose)


def normalize_point_timestamps(values: np.ndarray, frame_timestamp: int) -> np.ndarray:
    values = values.astype(np.float64)
    med = float(np.median(values))
    if med > 1e15:       # integer nanoseconds
        return np.rint(values).astype(np.int64)
    if med > 1e12:       # float nanoseconds
        return np.rint(values).astype(np.int64)
    if med > 1e6:        # absolute seconds
        return np.rint(values * 1e9).astype(np.int64)
    return np.rint(frame_timestamp + values * 1e9).astype(np.int64)


def transform_points_by_pose(
    xyz_imu_at_point: np.ndarray,
    point_timestamps: np.ndarray,
    pose_samples: list[tuple[int, dict]],
    target_timestamp: int,
) -> np.ndarray:
    target_inv = np.linalg.inv(interpolate_pose(pose_samples, target_timestamp))
    result = np.empty_like(xyz_imu_at_point, dtype=np.float64)
    # Quantizing only the interpolation request (not the point coordinates) keeps
    # this independent implementation practical while remaining far below pixel
    # resolution.  Raw point timestamps are normally shared by firing blocks.
    unique_ts, inverse = np.unique(point_timestamps, return_inverse=True)
    for i, ts in enumerate(unique_ts):
        mask = inverse == i
        relative = target_inv @ interpolate_pose(pose_samples, int(ts))
        result[mask] = xyz_imu_at_point[mask] @ relative[:3, :3].T + relative[:3, 3]
    return result


def project_raw_image(
    image: np.ndarray,
    xyz_imu: np.ndarray,
    lidar_ids: np.ndarray,
    T_imu_from_camera: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
    point_size: int,
) -> tuple[np.ndarray, int]:
    T_camera_from_imu = np.linalg.inv(T_imu_from_camera)
    xyz_camera = xyz_imu @ T_camera_from_imu[:3, :3].T + T_camera_from_imu[:3, 3]
    valid = xyz_camera[:, 2] > 0.3
    ids = np.flatnonzero(valid)
    uv = cv2.projectPoints(
        xyz_camera[ids].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, distortion
    )[0].reshape(-1, 2)
    h, w = image.shape[:2]
    inside = (
        np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    uv = np.rint(uv[inside]).astype(np.int32)
    point_ids = ids[inside]
    out = image.copy()
    colors = {1: (255, 180, 20), 2: (40, 220, 40), 14: (30, 70, 255)}
    radius = max(0, int(point_size) // 2)
    for (u, v), idx in zip(uv, point_ids):
        cv2.circle(out, (int(u), int(v)), radius, colors.get(int(lidar_ids[idx]), (255, 255, 255)), -1)
    return out, len(uv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--timestamp", required=True, type=int)
    ap.add_argument("--camera", default="camera1")
    ap.add_argument("--db", default="perception_experiment")
    ap.add_argument("--frame-collection", default="raw_data_frames_lidar14_0813")
    ap.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ap.add_argument("--raw-root", action="append", type=Path, dest="raw_roots")
    ap.add_argument("--pose-window-ms", type=float, default=250.0)
    ap.add_argument("--point-size", type=int, default=2)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    roots = args.raw_roots or DEFAULT_ROOTS
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10_000)
    frames = client[args.db][args.frame_collection]
    projection = {
        "timestamp": 1, "clip_id": 1, "bag_name": 1,
        f"dependency.sensors.{args.camera}": 1,
        "dependency.sensors.lidar_merge_nodeskew": 1,
        "dependency.ego_pose": 1,
    }
    frame = frames.find_one(
        {"clip_id": args.clip_id, "timestamp": args.timestamp}, projection
    )
    if frame is None:
        raise RuntimeError("raw frame not found")
    sensors = frame["dependency"]["sensors"]
    cam = sensors[args.camera]
    lidar = sensors["lidar_merge_nodeskew"]
    camera_ts = int(cam["timestamp"])
    lidar_ts = int(lidar["timestamp"])

    camera_path = content_path(cam["md5"], "camera", ".jpg", roots)
    lidar_path = content_path(lidar["md5"], "lidar", ".pcd", roots)
    image = cv2.imread(str(camera_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {camera_path}")
    pcd = read_binary_pcd(lidar_path)
    xyz = np.column_stack([pcd[k] for k in ("x", "y", "z")]).astype(np.float64)
    lidar_ids = np.asarray(pcd.get("lidar_id", np.zeros(len(xyz))), dtype=np.uint8)
    if "timestamp" not in pcd:
        raise ValueError("raw nodeskew PCD has no per-point timestamp")
    point_ts = normalize_point_timestamps(pcd["timestamp"], lidar_ts)

    margin = int(args.pose_window_ms * 1e6)
    pose_docs = frames.find(
        {"clip_id": args.clip_id, "timestamp": {"$gte": int(point_ts.min()) - margin,
                                                  "$lte": int(point_ts.max()) + margin}},
        {"timestamp": 1, "dependency.ego_pose": 1},
    ).sort("timestamp", 1)
    pose_samples = []
    for doc in pose_docs:
        ego = (doc.get("dependency") or {}).get("ego_pose") or {}
        pose = ego.get("pose")
        if pose:
            pose_samples.append((pose_stamp_ns(ego, int(doc["timestamp"])), pose))
    pose_samples.sort(key=lambda x: x[0])
    if len(pose_samples) < 2:
        raise RuntimeError("not enough raw ego poses for interpolation")

    xyz_at_camera = transform_points_by_pose(xyz, point_ts, pose_samples, camera_ts)
    K = np.asarray(cam["intrinsic"]["intrinsic"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(cam["intrinsic"]["distortion"], dtype=np.float64).reshape(-1)
    T_imu_from_camera = np.asarray(cam["extrinsic"]["transformation"], dtype=np.float64).reshape(4, 4)

    raw_overlay, n_raw = project_raw_image(
        image, xyz, lidar_ids, T_imu_from_camera, K, distortion, args.point_size
    )
    compensated_overlay, n_comp = project_raw_image(
        image, xyz_at_camera, lidar_ids, T_imu_from_camera, K, distortion, args.point_size
    )
    cv2.putText(raw_overlay, "RAW nodeskew / no ego-time compensation", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(compensated_overlay, "RAW nodeskew / per-point pose -> camera timestamp", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_out = args.output_dir / "raw_no_time_compensation.jpg"
    comp_out = args.output_dir / "raw_per_point_to_camera_time.jpg"
    side_out = args.output_dir / "comparison.jpg"
    cv2.imwrite(str(raw_out), raw_overlay)
    cv2.imwrite(str(comp_out), compensated_overlay)
    cv2.imwrite(str(side_out), np.hstack([raw_overlay, compensated_overlay]))

    provenance = {
        "db": args.db, "frame_collection": args.frame_collection,
        "clip_id": args.clip_id, "frame_timestamp": args.timestamp,
        "bag_name": frame.get("bag_name"), "camera": args.camera,
        "camera_timestamp": camera_ts, "lidar_timestamp": lidar_ts,
        "camera_minus_lidar_ms": (camera_ts - lidar_ts) / 1e6,
        "camera": {"md5": cam["md5"], "path": str(camera_path),
                   "verified_md5": md5_file(camera_path)},
        "lidar": {"sensor": "lidar_merge_nodeskew", "md5": lidar["md5"],
                  "path": str(lidar_path), "verified_md5": md5_file(lidar_path),
                  "points": len(xyz), "point_timestamp_min": int(point_ts.min()),
                  "point_timestamp_max": int(point_ts.max())},
        "pose": {"samples": len(pose_samples), "timestamp_min": pose_samples[0][0],
                 "timestamp_max": pose_samples[-1][0]},
        "calibration": {"K": K.tolist(), "distortion": distortion.tolist(),
                        "T_imu_from_camera": T_imu_from_camera.tolist()},
        "projected_points": {"raw_no_time_compensation": n_raw,
                             "per_point_to_camera_time": n_comp},
        "outputs": {"raw": str(raw_out), "compensated": str(comp_out),
                    "comparison": str(side_out)},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
