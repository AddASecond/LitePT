#!/usr/bin/env python3
"""REJECT clip 聚合点云 BEV 可视化（供人工查验 badcase）。

每 clip 取 30 帧（均匀跨全 clip），按各自 ego_pose 变换到 map 系聚合，
BEV 俯视着色 = 帧序号（紫=早 黄=晚）。pose 错的 clip 会看到同一结构
被多帧复制到不同位置（彩虹分层）；pose 正常的 clip 结构是单色坍缩。

输出:
  exp/robotruck/reject_bev/reject_<i>_<clip8>.png   单 clip 大图
  exp/robotruck/reject_bev/contact_sheet.png        23 格总拼图
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

LIST = ROOT / "exp/robotruck/pose_badcase/final_badcase_list.txt"
OUTDIR = ROOT / "exp/robotruck/reject_bev"
N_SAMPLE = 30
SUBSAMPLE = 12000


def _load_layer_scan():
    spec = importlib.util.spec_from_file_location("layer_scan", ROOT / "tools/layer_scan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reject_clips() -> list[tuple[str, str]]:
    rows = []
    for line in LIST.read_text().splitlines():
        cols = line.split("\t")
        if cols and cols[0] == "REJECT":
            rows.append((cols[1], cols[2]))
    return rows


def collect_clip(ls, cid: str):
    recs = ls.fetch_records([cid])
    fr = [f for f in recs[0]["frames"] if f[1] is not None and f[3] is not None]
    if len(fr) < 5:
        return None
    idx = np.unique(np.linspace(0, len(fr) - 1, min(N_SAMPLE, len(fr))).astype(int))
    pts, fids, traj = [], [], []
    for k, i in enumerate(idx):
        ts, t, R, md5 = fr[i]
        try:
            xyz = ls.load_cloud(cid, ts, md5)
        except Exception:
            continue
        if len(xyz) > SUBSAMPLE:
            xyz = xyz[:: len(xyz) // SUBSAMPLE]
        pts.append(xyz @ R.T + t)
        fids.append(np.full(len(xyz), k))
        traj.append(t)
    if len(pts) < 5:
        return None
    return np.concatenate(pts), np.concatenate(fids), np.array(traj), len(idx)


def render_one(ax, data, title: str) -> None:
    if data is None:
        ax.set_title(title + "\n(no data)")
        ax.axis("off")
        return
    pts, fids, traj, n_used = data
    # 视野以轨迹 2-98% 分位为中心
    lo = np.percentile(traj, 2, axis=0)
    hi = np.percentile(traj, 98, axis=0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + 60
    ax.scatter(pts[:, 0], pts[:, 1], c=fids, cmap="viridis", s=0.6, linewidths=0)
    # 轨迹按帧序渐变
    segs = np.stack([traj[:-1, :2], traj[1:, :2]], axis=1)
    lc = LineCollection(segs, cmap="viridis",
                        array=np.arange(len(segs)), linewidths=2)
    ax.add_collection(lc)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main() -> int:
    ls = _load_layer_scan()
    clips = reject_clips()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(clips)} REJECT clips", flush=True)
    data_all = []
    for i, (cid, reason) in enumerate(clips):
        try:
            data = collect_clip(ls, cid)
        except Exception as exc:
            print(f"[{i+1}/{len(clips)}] {cid[:13]} collect error: {exc}", flush=True)
            data = None
        data_all.append((cid, reason, data))
        fig, ax = plt.subplots(figsize=(9, 9))
        render_one(ax, data, f"{cid}\n{reason}  ({data[3] if data else 0} frames)")
        fig.tight_layout()
        p = OUTDIR / f"reject_{i+1:02d}_{cid[:8]}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"[{i+1}/{len(clips)}] {cid[:13]} -> {p.name}", flush=True)
    # 总拼图
    import math
    cols = 5
    rows = math.ceil(len(clips) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6))
    for ax in np.ravel(axes)[len(clips):]:
        ax.axis("off")
    for (cid, reason, data), ax in zip(data_all, np.ravel(axes)):
        render_one(ax, data, f"{cid[:13]} {reason.split(':')[0]}")
    fig.suptitle("REJECT clips aggregate BEV (color = frame order early->late)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "contact_sheet.png", dpi=110)
    plt.close(fig)
    print("DONE ->", OUTDIR / "contact_sheet.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
