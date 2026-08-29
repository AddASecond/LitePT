#!/usr/bin/env python3
"""Validate a deskew PCD against nodeskew point timestamps and ego poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = {}
        while True:
            line = stream.readline().decode("ascii").strip()
            words = line.split()
            if words and not line.startswith("#"):
                header[words[0]] = words[1:]
            if line.startswith("DATA "):
                break
        mapping = {
            ("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1",
            ("U", 2): "<u2", ("U", 4): "<u4", ("I", 8): "<i8",
        }
        dtype = np.dtype([
            (name, mapping[(kind, int(size))])
            for name, kind, size in zip(
                header["FIELDS"], header["TYPE"], header["SIZE"]
            )
        ])
        return np.frombuffer(stream.read(), dtype=dtype).copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodeskew", required=True, type=Path)
    ap.add_argument("--deskew", required=True, type=Path)
    ap.add_argument("--cache-frames", required=True, type=Path)
    ap.add_argument("--reference-timestamp", required=True, type=int)
    args = ap.parse_args()
    raw = read_pcd(args.nodeskew)
    deskew = read_pcd(args.deskew)
    rows = []
    for path in args.cache_frames.glob("*/frame.json"):
        doc = json.loads(path.read_text())
        ego = (doc.get("dependency") or {}).get("ego_pose")
        if not ego or not ego.get("pose"):
            continue
        stamp = ego["header"]["stamp"]
        ts = int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])
        p = ego["pose"]
        rows.append((
            ts,
            [p["position"][k] for k in ("x", "y", "z")],
            [p["orientation"][k] for k in ("x", "y", "z", "w")],
        ))
    rows.sort()
    pose_t = np.asarray([r[0] for r in rows], np.float64)
    pose_p = np.asarray([r[1] for r in rows], np.float64)
    pose_q = np.asarray([r[2] for r in rows], np.float64)
    for i in range(1, len(pose_q)):
        if np.dot(pose_q[i - 1], pose_q[i]) < 0:
            pose_q[i] *= -1

    def interpolate(query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qf = np.asarray(query_ns, np.float64)
        pos = np.column_stack([np.interp(qf, pose_t, pose_p[:, j]) for j in range(3)])
        quat = np.column_stack([np.interp(qf, pose_t, pose_q[:, j]) for j in range(4)])
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        return pos, quat

    xyz = np.column_stack([raw[k] for k in ("x", "y", "z")]).astype(np.float64)
    xyz_d = np.column_stack([deskew[k] for k in ("x", "y", "z")]).astype(np.float64)
    pos, quat = interpolate(raw["timestamp"])
    ref_pos, ref_quat = interpolate(np.array([args.reference_timestamp]))
    xyz_map = Rotation.from_quat(quat).apply(xyz) + pos
    expected = Rotation.from_quat(ref_quat[0]).inv().apply(xyz_map - ref_pos[0])
    residual = np.linalg.norm(expected - xyz_d, axis=1)
    by_lidar = {}
    for lidar_id in np.unique(raw["lidar_id"]):
        selected = residual[raw["lidar_id"] == lidar_id]
        by_lidar[str(int(lidar_id))] = {
            "n": int(len(selected)),
            "p50_m": float(np.percentile(selected, 50)),
            "p90_m": float(np.percentile(selected, 90)),
            "within_2cm": float(np.mean(selected < 0.02)),
        }
    print(json.dumps({
        "n": int(len(residual)),
        "residual_m_percentiles": dict(zip(
            ("p0", "p50", "p90", "p95", "p99", "p100"),
            np.percentile(residual, [0, 50, 90, 95, 99, 100]).tolist(),
        )),
        "within_2cm": float(np.mean(residual < 0.02)),
        "within_10cm": float(np.mean(residual < 0.10)),
        "by_lidar": by_lidar,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
