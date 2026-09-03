#!/usr/bin/env python3
"""Pose-layer badcase 扫描（判别聚合鬼影/点云分层的 pose 根因）。

只读 mongo frames 的 ego_pose，不做点云聚合（速度优先）。
指标（帧间隔 dt≈0.1s）：
  vrate   垂直速度 |Δz|/dt          —— 卡车不可能瞬时垂直跳变
  accel   位置二阶差分 |p[i+1]-2p[i]+p[i-1]|/dt² —— pose 跳变/横向振荡
  speed   位移速度 |Δp|/dt           —— p95 vs median 稳定性
Badcase 判别（任一命中）：
  Z_JUMP         max_vrate > 5 m/s
  TELEPORT       max_accel > 50 m/s²
  SPEED_INSTABLE p95_speed > 2*median_speed + 2 m/s
  MISSING_POSE   缺 pose 比例 > 0.1

用法:
  tools/pose_badcase_scan.py --clips <id> [<id> ...]      # 验证指定 clip
  tools/pose_badcase_scan.py --all                        # 全量扫描 clips 集合
输出: exp/robotruck/pose_badcase/{metrics.json, badcase_list.txt}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/robotruck/pose_badcase"

MONGO_URI = os.environ.get("ROBOTRUCK_MONGO_URI",
                           "mongodb://krk030-mongodb:27017/?authSource=perception_experiment")
DB = "perception_experiment"
FRAMES_COL = "raw_data_frames_lidar14_0813"
CLIPS_COL = "raw_data_clips_lidar14_0813"

MIN_FRAMES = 30
VRATE_MAX = 5.0       # m/s
ACCEL_MAX = 50.0      # m/s^2
SPEED_SPIKE = 2.0     # factor
SPEED_MARGIN = 2.0    # m/s
MISSING_RATIO = 0.1


def clip_metrics(clip_id: str, frames_col) -> dict:
    cur = frames_col.find(
        {"clip_id": clip_id},
        {"timestamp": 1, "dependency.ego_pose.pose": 1},
    ).sort("timestamp", 1)
    ts, xyz = [], []
    n_missing = 0
    for doc in cur:
        ts.append(int(doc.get("timestamp") or 0))
        pose = (((doc.get("dependency") or {}).get("ego_pose") or {}).get("pose"))
        if pose and "position" in pose:
            p = pose["position"]
            xyz.append((float(p["x"]), float(p["y"]), float(p["z"])))
        else:
            n_missing += 1
            xyz.append(None)
    m = {"clip_id": clip_id, "n_frames": len(ts)}
    if len(ts) < MIN_FRAMES:
        m["flags"] = ["TOO_SHORT"] if len(ts) else ["NO_FRAMES"]
        return m
    m["missing_ratio"] = round(n_missing / len(ts), 4)
    valid = np.array([p for p in xyz if p is not None], np.float64)
    t = np.array([x for x, p in zip(ts, xyz) if p is not None], np.int64)
    if len(valid) < MIN_FRAMES:
        m["flags"] = ["NO_POSE"]
        return m
    dt = np.diff(t).astype(np.float64) / 1e9
    dt = np.where(dt <= 0, np.nan, dt)
    dp = np.diff(valid, axis=0)
    dist = np.linalg.norm(dp, axis=1)
    with np.errstate(invalid="ignore"):
        speed = dist / dt
        vrate = np.abs(dp[:, 2]) / dt
        # 位置二阶差分（相邻三帧），dt 用两侧 dt 之积近似 dt^2
        d2 = valid[2:] - 2 * valid[1:-1] + valid[:-2]
        dt2 = dt[1:] * dt[:-1]
        accel = np.linalg.norm(d2, axis=1) / dt2
    m.update({
        "median_speed": round(float(np.nanmedian(speed)), 3),
        "p95_speed": round(float(np.nanpercentile(speed, 95)), 3),
        "max_speed": round(float(np.nanmax(speed)), 3),
        "max_vrate": round(float(np.nanmax(vrate)), 3),
        "max_accel": round(float(np.nanmax(accel)), 2),
        "p99_accel": round(float(np.nanpercentile(accel, 99)), 2),
        "z_range": round(float(valid[:, 2].max() - valid[:, 2].min()), 3),
    })
    flags = []
    if m["missing_ratio"] > MISSING_RATIO:
        flags.append("MISSING_POSE")
    if m["max_vrate"] > VRATE_MAX:
        flags.append("Z_JUMP")
    if m["max_accel"] > ACCEL_MAX:
        flags.append("TELEPORT")
    if m["p95_speed"] > SPEED_SPIKE * m["median_speed"] + SPEED_MARGIN:
        flags.append("SPEED_INSTABLE")
    m["flags"] = flags
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    from pymongo import MongoClient
    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    frames = c[DB][FRAMES_COL]

    if args.all:
        clip_ids = [d["clip_id"] for d in
                    c[DB][CLIPS_COL].find({}, {"clip_id": 1}) if d.get("clip_id")]
        print(f"total clips: {len(clip_ids)}", flush=True)
    else:
        clip_ids = args.clips

    metrics = []
    for i, cid in enumerate(clip_ids):
        try:
            m = clip_metrics(cid, frames)
        except Exception as exc:
            m = {"clip_id": cid, "flags": [f"ERROR:{type(exc).__name__}"], "error": str(exc)}
        metrics.append(m)
        if (i + 1) % 50 == 0 or i + 1 == len(clip_ids):
            print(f"  scanned {i + 1}/{len(clip_ids)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    bad = [m for m in metrics if m.get("flags")]
    good = [m for m in metrics if not m.get("flags")]
    (OUT / "badcase_list.txt").write_text(
        "\n".join(f"{m['clip_id']}\t{','.join(m['flags'])}" for m in bad))
    # 汇总
    flag_count: dict = defaultdict(int)
    for m in bad:
        for f in m["flags"]:
            flag_count[f.split(":")[0]] += 1
    print("\n===== SUMMARY =====")
    print(f"total={len(metrics)} good={len(good)} bad={len(bad)}")
    print("flags:", dict(sorted(flag_count.items(), key=lambda kv: -kv[1])))
    print("\nbadcase list (first 50):")
    for m in bad[:50]:
        print(f"  {m['clip_id'][:13]}  {','.join(m['flags'])}  "
              f"vrate={m.get('max_vrate')} accel={m.get('max_accel')} "
              f"p95v={m.get('p95_speed')} medv={m.get('median_speed')}")
    print(f"\nfull: {OUT / 'badcase_list.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
