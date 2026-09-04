#!/usr/bin/env python3
"""Offline PC layer helpers + scan for pose_badcase (global_pc_p20 / mongo loaders).

Delivery reject SoT is quality_gate.py, not this file.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from paths import ROOT, ensure_import_path

ensure_import_path()
import store as STORE
from validate_projection import (  # noqa: E402
    read_binary_pcd, quat_to_rotation)

BACKUP = ROOT / "exp/robotruck/raw_volume_cache"
OUT = ROOT / "exp/robotruck/pose_badcase"
RAW_ROOTS = list(STORE.RAW_ROOTS)

MONGO_URI = STORE.DEFAULT_URI
DB = "perception_experiment"
FRAMES_COL = "raw_data_frames_lidar14_0813"
CLIPS_COL = "raw_data_clips_lidar14_0813"

# pose 阈值
VRATE_MAX = 5.0
ACCEL_MAX = 50.0
SPEED_SPIKE, SPEED_MARGIN = 2.0, 2.0
MISSING_RATIO = 0.1
# 点云分层阈值（标定后可调）
PC_LAYER_MAX = 0.35   # p20 NN 跨帧中位数 (m)

SUBSAMPLE = 20000
PAIR_GAP = 10         # 帧间隔 (≈1s/20m，保证两帧视野大量重叠)


def load_cloud(clip_uuid: str, ts: str, lidar_md5: str) -> np.ndarray:
    """(N,3) vehicle-frame xyz; 本地 bin 优先, 否则 md5 PCD 直读。"""
    fr = BACKUP / f"batch_s5_{clip_uuid}" / "frames" / str(ts)
    binp = fr / "lidar_merge.bin"
    if binp.is_file():
        arr = np.fromfile(binp, dtype=np.float32).reshape(-1, 7)
        return arr[:, :3].astype(np.float64)
    rel = Path("lidar") / lidar_md5[:2] / lidar_md5[2:4] / f"{lidar_md5[4:]}.pcd"
    for r in RAW_ROOTS:
        p = r / rel
        if p.is_file():
            d = read_binary_pcd(p)
            return np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64)
    raise FileNotFoundError(f"no lidar payload ts={ts}")


def pc_pair_metric(pa: np.ndarray, pb: np.ndarray) -> float:
    """两帧点云（已变换到 map 系）的跨帧对齐度: p20 of NN dist."""
    if len(pa) > SUBSAMPLE:
        pa = pa[:: len(pa) // SUBSAMPLE]
    if len(pb) > SUBSAMPLE:
        pb = pb[:: len(pb) // SUBSAMPLE]
    from scipy.spatial import cKDTree
    tree = cKDTree(pa)
    d, _ = tree.query(pb, k=1, workers=-1)
    return float(np.percentile(d, 20))


def scan_clip(rec: dict) -> dict:
    """rec: {clip_id, frames: [(ts, xyz, R|None), ...]}"""
    cid = rec["clip_id"]
    fr = rec["frames"]
    m = {"clip_id": cid, "n_frames": len(fr)}

    # ---- pose 指标 ----
    t = np.array([f[0] for f in fr], np.int64)
    P = np.array([f[1] for f in fr], np.float64)
    dt = np.diff(t).astype(np.float64) / 1e9
    dt = np.where(dt <= 0, np.nan, dt)
    dp = np.diff(P, axis=0)
    dist = np.linalg.norm(dp, axis=1)
    with np.errstate(invalid="ignore"):
        speed = dist / dt
        vrate = np.abs(dp[:, 2]) / dt
        d2 = P[2:] - 2 * P[1:-1] + P[:-2]
        accel = np.linalg.norm(d2, axis=1) / (dt[1:] * dt[:-1])
    m.update({
        "median_speed": round(float(np.nanmedian(speed)), 3),
        "p95_speed": round(float(np.nanpercentile(speed, 95)), 3),
        "max_vrate": round(float(np.nanmax(vrate)), 3),
        "max_accel": round(float(np.nanmax(accel)), 2),
    })
    flags = []
    if m["max_vrate"] > VRATE_MAX:
        flags.append("POSE_Z_JUMP")
    if m["max_accel"] > ACCEL_MAX:
        flags.append("POSE_TELEPORT")
    if m["p95_speed"] > SPEED_SPIKE * m["median_speed"] + SPEED_MARGIN:
        flags.append("POSE_SPEED_INSTABLE")

    # ---- 点云分层指标 ----
    n = len(fr)
    if n < 10:
        m["pc_p20"] = None
        m["pc_flags"] = ["TOO_SHORT"]
        m["flags"] = flags + m["pc_flags"]
        return m
    idx = sorted(set([0, n // 2, n - 1]))
    idx = [i for i in idx if fr[i][2] is not None and fr[i][3] is not None]
    pair_vals = []
    for a, b in zip(idx[:-1], idx[1:]):
        b = min(a + PAIR_GAP, n - 1) if len(idx) > 1 else a
        if b <= a or fr[b][2] is None or fr[b][3] is None:
            continue
        try:
            A = load_cloud(cid, fr[a][0], fr[a][3])
            B = load_cloud(cid, fr[b][0], fr[b][3])
        except Exception as exc:
            m["pc_error"] = str(exc)[:120]
            continue
        Ra, Rb = fr[a][2], fr[b][2]
        ta = P[a] if Ra is not None else None
        tb = P[b] if Rb is not None else None
        if ta is None or tb is None:
            continue
        pa = A @ Ra.T + ta
        pb = B @ Rb.T + tb
        pair_vals.append(pc_pair_metric(pa, pb))
    m["pc_p20"] = round(float(np.median(pair_vals)), 3) if pair_vals else None
    pc_flags = []
    if m["pc_p20"] is not None and m["pc_p20"] > PC_LAYER_MAX:
        pc_flags.append("PC_LAYER")
    m["pc_flags"] = pc_flags
    m["flags"] = flags + pc_flags
    return m


def fetch_records(clip_ids: list[str]) -> list[dict]:
    from pymongo import MongoClient
    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    frames = c[DB][FRAMES_COL]
    recs = []
    for i, cid in enumerate(clip_ids):
        fr = []
        for doc in frames.find(
                {"clip_id": cid},
                {"timestamp": 1, "dependency.ego_pose.pose": 1,
                 "dependency.sensors.lidar_merge_deskew.md5": 1,
                 "dependency.sensors.lidar_merge_nodeskew.md5": 1,
                 "dependency.sensors.lidar_merge.md5": 1},
        ).sort("timestamp", 1):
            pose = (((doc.get("dependency") or {}).get("ego_pose") or {}).get("pose"))
            if not pose or "position" not in pose:
                fr.append((int(doc.get("timestamp") or 0), None, None, None))
                continue
            R = quat_to_rotation(pose["orientation"])
            t = np.array([float(pose["position"][k]) for k in ("x", "y", "z")])
            sensors = ((doc.get("dependency") or {}).get("sensors") or {})
            md5 = None
            for k in ("lidar_merge_deskew", "lidar_merge_nodeskew", "lidar_merge"):
                s = sensors.get(k) or {}
                if s.get("md5") and len(s["md5"]) == 32:
                    md5 = s["md5"]
                    break
            fr.append((int(doc.get("timestamp") or 0), t, R, md5))
        recs.append({"clip_id": cid, "frames": fr})
        if (i + 1) % 100 == 0:
            print(f"  fetched {i + 1}/{len(clip_ids)}", flush=True)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from pymongo import MongoClient
    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    if args.all:
        clip_ids = [d["clip_id"] for d in
                    c[DB][CLIPS_COL].find({}, {"clip_id": 1}) if d.get("clip_id")]
    else:
        clip_ids = args.clips
    print(f"clips: {len(clip_ids)}", flush=True)

    recs = fetch_records(clip_ids)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        metrics = list(ex.map(scan_clip, recs))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layer_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    bad = [m for m in metrics if m.get("flags")]
    (OUT / "layer_badcase_list.txt").write_text(
        "\n".join(f"{m['clip_id']}\t{','.join(m['flags'])}"
                  f"\tmax_accel={m.get('max_accel')}\tmax_vrate={m.get('max_vrate')}"
                  f"\tpc_p20={m.get('pc_p20')}" for m in bad))
    from collections import defaultdict
    fc = defaultdict(int)
    for m in bad:
        for f in m["flags"]:
            fc[f.split(":")[0]] += 1
    print("\n===== SUMMARY =====")
    print(f"total={len(metrics)} bad={len(bad)} good={len(metrics) - len(bad)}")
    print("flags:", dict(sorted(fc.items(), key=lambda kv: -kv[1])))
    print("\nbadcase (first 60):")
    for m in bad[:60]:
        print(f"  {m['clip_id'][:13]}  {','.join(m['flags'])}  "
              f"accel={m.get('max_accel')} vrate={m.get('max_vrate')} "
              f"pc_p20={m.get('pc_p20')}")
    print(f"\nfull list: {OUT / 'layer_badcase_list.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
