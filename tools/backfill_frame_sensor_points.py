#!/usr/bin/env python3
"""Add unfiltered per-frame deskew point assets to an existing OCC scene."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np

import export_robotruck_occ_scene as exporter


def _pose_stamp_ns(ego: dict) -> int:
    stamp = ego["header"]["stamp"]
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])


def _interpolate_pose_matrix(samples: list[tuple[int, dict]], timestamp_ns: int) -> np.ndarray:
    times = [row[0] for row in samples]
    hi = bisect.bisect_left(times, int(timestamp_ns))
    if hi <= 0:
        return exporter.sag.ego_pose_to_T_map_vehicle(samples[0][1])
    if hi >= len(samples):
        return exporter.sag.ego_pose_to_T_map_vehicle(samples[-1][1])
    ta, pa = samples[hi - 1]
    tb, pb = samples[hi]
    alpha = float(timestamp_ns - ta) / float(tb - ta)
    posa, posb = pa["position"], pb["position"]
    qa = np.array([pa["orientation"][k] for k in ("x", "y", "z", "w")], np.float64)
    qb = np.array([pb["orientation"][k] for k in ("x", "y", "z", "w")], np.float64)
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = (1.0 - alpha) * qa + alpha * qb
    q /= np.linalg.norm(q)
    pose = {
        "position": {
            k: (1.0 - alpha) * float(posa[k]) + alpha * float(posb[k])
            for k in ("x", "y", "z")
        },
        "orientation": dict(zip(("x", "y", "z", "w"), q.tolist())),
    }
    return exporter.sag.ego_pose_to_T_map_vehicle(pose)
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--cache-root", default="exp/robotruck/raw_volume_cache")
    ap.add_argument("--scene-root", default="exp/robotruck/occ_scenes")
    ap.add_argument("--pred-root", default="exp/robotruck/clip_video")
    args = ap.parse_args()

    scene = Path(args.scene_root) / args.clip
    cache = Path(args.cache_root) / args.clip
    pred_dir = Path(args.pred_root) / args.clip / "preds"
    index = json.loads((scene / "index.json").read_text())
    pose_samples = []
    for frame_json in (cache / "frames").glob("*/frame.json"):
        doc = json.loads(frame_json.read_text())
        ego = (doc.get("dependency") or {}).get("ego_pose")
        if ego and ego.get("pose") and ego.get("header", {}).get("stamp"):
            pose_samples.append((_pose_stamp_ns(ego), ego["pose"]))
    pose_samples.sort(key=lambda row: row[0])
    if len(pose_samples) < 2:
        raise ValueError("at least two timestamped ego poses are required")
    done = 0
    agg_xyz: list[np.ndarray] = []
    agg_labels: list[np.ndarray] = []
    agg_lidar: list[np.ndarray] = []
    for entry in index["frames"]:
        ts = str(entry.get("timestamp") or entry["frame_id"])
        pts = exporter._h.load_lidar_bin(
            cache / "frames" / ts / "lidar_merge.bin",
            num_cols=len(exporter._h.LIDAR_COLS),
        )
        pred = np.load(pred_dir / f"{ts}_pred.npy").astype(np.uint8).reshape(-1)
        if pred.shape[0] != pts.shape[0]:
            raise ValueError(f"{ts}: pred={pred.shape[0]} points={pts.shape[0]}")
        xyz = pts[:, :3].astype(np.float32)
        lidar = pts[:, 6].astype(np.uint8)
        frame_doc = json.loads((cache / "frames" / ts / "frame.json").read_text())
        pose = ((frame_doc.get("dependency") or {}).get("ego_pose") or {}).get("pose")
        if not pose:
            raise ValueError(f"{ts}: dependency.ego_pose.pose is missing")
        xyz_map = exporter.sag.transform_points(
            xyz, exporter.sag.ego_pose_to_T_map_vehicle(pose)
        )
        agg_xyz.append(xyz_map)
        agg_labels.append(pred)
        agg_lidar.append(lidar)
        frame_dir = scene / "frames" / ts
        xyz.tofile(frame_dir / "frame_sensor_points_xyz.f32.bin")
        pred.tofile(frame_dir / "frame_sensor_points_labels.u8.bin")
        lidar.tofile(frame_dir / "frame_sensor_points_lidar_id.u8.bin")
        meta_path = frame_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        sensors = ((frame_doc.get("dependency") or {}).get("sensors") or {})
        lidar_doc = sensors.get("lidar_merge_deskew") or {}
        lidar_timestamp = int(lidar_doc.get("timestamp") or ts)
        t_map_v_lidar = _interpolate_pose_matrix(pose_samples, lidar_timestamp)
        for camera in meta.get("cameras", []):
            camera_doc = sensors.get(camera["name"]) or {}
            camera_timestamp = camera_doc.get("timestamp")
            if camera_timestamp is None:
                continue
            t_map_v_camera = _interpolate_pose_matrix(
                pose_samples, int(camera_timestamp)
            )
            t_c_v = np.asarray(camera["T_c_v"], np.float64).reshape(4, 4)
            t_c_v_lidar = (
                t_c_v @ np.linalg.inv(t_map_v_camera) @ t_map_v_lidar
            )
            camera["T_c_v_lidar_ref"] = t_c_v_lidar.reshape(-1).tolist()
            camera["time_compensation"] = {
                "method": "ego_pose_linear_position_nlerp_quaternion",
                "lidar_reference_timestamp": lidar_timestamp,
                "camera_timestamp": int(camera_timestamp),
                "delta_ms": (int(camera_timestamp) - lidar_timestamp) / 1e6,
            }
        prefix = f"frames/{ts}"
        n = int(xyz.shape[0])
        meta.setdefault("assets", {})["frame_sensor_points"] = {
            "n": n,
            "source": "lidar_merge_deskew",
            "filtering": "none",
            "xyz": exporter.asset_ref(f"{prefix}/frame_sensor_points_xyz.f32.bin", "float32", [n, 3]),
            "labels": exporter.asset_ref(f"{prefix}/frame_sensor_points_labels.u8.bin", "uint8", [n]),
            "lidar_id": exporter.asset_ref(f"{prefix}/frame_sensor_points_lidar_id.u8.bin", "uint8", [n]),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        done += 1
    xyz_all = np.concatenate(agg_xyz, axis=0)
    labels_all = np.concatenate(agg_labels, axis=0)
    lidar_all = np.concatenate(agg_lidar, axis=0)
    agg_dir = scene / "point_aggregate"
    agg_dir.mkdir(exist_ok=True)
    xyz_all.astype(np.float32, copy=False).tofile(agg_dir / "xyz_map.f32.bin")
    labels_all.astype(np.uint8, copy=False).tofile(agg_dir / "labels.u8.bin")
    lidar_all.astype(np.uint8, copy=False).tofile(agg_dir / "lidar_id.u8.bin")
    n_all = int(xyz_all.shape[0])
    index["point_aggregate"] = {
        "n": n_all,
        "n_frames": done,
        "source": "lidar_merge_deskew",
        "filtering": "none",
        "sampling": "none",
        "voxelization": "none",
        "coordinate": "map",
        "xyz_map": exporter.asset_ref("point_aggregate/xyz_map.f32.bin", "float32", [n_all, 3]),
        "labels": exporter.asset_ref("point_aggregate/labels.u8.bin", "uint8", [n_all]),
        "lidar_id": exporter.asset_ref("point_aggregate/lidar_id.u8.bin", "uint8", [n_all]),
    }
    (scene / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps({
        "clip": args.clip,
        "frames": done,
        "points": n_all,
        "source": "lidar_merge_deskew",
        "filtering": "none",
        "sampling": "none",
        "voxelization": "none",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
