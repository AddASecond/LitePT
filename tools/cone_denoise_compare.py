#!/usr/bin/env python3
"""去噪前后对比 PNG：BEFORE (红=保留, 蓝=被剔除) vs AFTER (红=保留)。

布局与已确认的 cone_frame_preview 相同（BEV + 7 相机），上下两组对比。
投影完全复用 render_robotruck_clip_video.py 管线（parse_camera /
camera_time_compensated_T / project_points / draw_projection）。

用法:
  .venv_smoke/bin/python tools/cone_denoise_compare.py            # 全部 7 case
  .venv_smoke/bin/python tools/cone_denoise_compare.py --clip <clip_id>
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "exp/robotruck/raw_volume_cache"
MID = ROOT / "exp/robotruck/cone_mid/denoised"
OUT = ROOT / "exp/robotruck/cone_png"
PANEL_W, PANEL_H = 800, 450

GOOD_CLIPS = [
    "batch_s5_11c2aa2d-2618-45d8-ab28-5cf1529eca84",
    "batch_s5_6bc60101-e684-438b-9983-b74c1cc5fe2b",
    "batch_s5_f1683bae-a2a6-496a-b9a7-9cc2ccfbc3e0",
    "batch_s5_72ebcd30-693f-417e-855d-ae66447d2020",
    "batch_s5_9d17f68b-3594-4412-b39e-deb7e7250de9",
    "batch_s5_701ff460-2b29-4433-b492-39e7ae2310b6",
    "batch_s5_c5d9f4b4-e410-4241-8e46-f5653db90f5b",
]

RED_BGR = (0, 0, 255)      # kept
BLUE_BGR = (255, 80, 0)    # removed


def load_render_module():
    path = ROOT / "tools" / "render_robotruck_clip_video.py"
    spec = importlib.util.spec_from_file_location("rrv_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rrv = load_render_module()


def pick_best_frame(data: dict) -> dict:
    """选 n_in 最大的帧。"""
    best = None
    for fr in data["per_frame"]:
        n = fr.get("denoise", {}).get("n_in", 0)
        if best is None or n > best.get("denoise", {}).get("n_in", 0):
            best = fr
    return best


def bev_panel(pts_kept, pts_removed, ts: str, tag: str) -> np.ndarray:
    """RGB 输出。"""
    fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    if pts_removed is not None and len(pts_removed):
        ax.scatter(pts_removed[:, 0], pts_removed[:, 1], s=4, c="#4a90d9",
                   linewidths=0)
    if len(pts_kept):
        ax.scatter(pts_kept[:, 0], pts_kept[:, 1], s=5, c="red", linewidths=0)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-15, 90)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("white")
    n_k = len(pts_kept)
    n_r = 0 if pts_removed is None else len(pts_removed)
    ax.set_title(f"{tag} BEV  ts={ts}\nred=kept({n_k}) blue=removed({n_r})",
                 fontsize=8, color="white")
    fig.patch.set_facecolor("black")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def cam_panel(img_bgr, overlays, cam_doc, pose_samples, lidar_ts,
              cam_name: str) -> np.ndarray:
    """锥桶点集投影 —— 复用 rrv 投影链。overlays: [(pts, color_bgr, radius), ...]"""
    K, dist5, T_c_v, cal_w, cal_h = rrv.parse_camera(cam_doc)
    cam_ts = int(cam_doc.get("timestamp") or int(lidar_ts))
    T_c_v_proj = rrv.camera_time_compensated_T(T_c_v, pose_samples, lidar_ts, cam_ts)
    ih, iw = img_bgr.shape[:2]
    if iw != cal_w or ih != cal_h:
        K = K.copy()
        K[0, :] *= iw / float(cal_w)
        K[1, :] *= ih / float(cal_h)

    vis = img_bgr.copy()
    for pts, color_bgr, radius in overlays:
        if len(pts) == 0:
            continue
        uv, _ = rrv.project_points(
            pts.astype(np.float64),
            np.zeros(len(pts), np.int32),      # dummy labels（颜色由 cols 覆盖）
            K, dist5, T_c_v_proj, iw, ih,
            max_points=400000, seed=0,
        )
        cols = np.zeros((uv.shape[0], 3), np.uint8)
        cols[:] = color_bgr
        vis = rrv.draw_projection(vis, uv, cols, radius=radius, alpha=1.0)

    vis = cv2.resize(vis, (PANEL_W, PANEL_H))
    n_pts = sum(len(p) for p, _, _ in overlays)
    cv2.putText(vis, f"{cam_name} pts={n_pts}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def frame_panels(fr: dict, clip_dir: Path, ts: str, pose_samples, tag: str) -> list:
    """一组 8 面板（BGR）：BEV + 7 相机。"""
    xyz_all = fr["cone_points_xyz"].astype(np.float32)
    xyz_kept = fr.get("cone_points_xyz_denoised")
    if xyz_kept is None:
        xyz_kept = xyz_all
    xyz_kept = xyz_kept.astype(np.float32)
    if len(xyz_all) == len(xyz_kept):
        pts_removed = np.zeros((0, 3), np.float32)
    else:
        tree = cKDTree(xyz_kept)
        d, _ = tree.query(xyz_all, k=1, workers=-1)
        pts_removed = xyz_all[d > 1e-6]

    panels = [cv2.cvtColor(bev_panel(xyz_kept, pts_removed, ts, tag),
                           cv2.COLOR_RGB2BGR)]
    meta = rrv.json.loads((clip_dir / "frames" / ts / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]
    for cam_name in rrv.CAM_ORDER:
        img_path = clip_dir / "frames" / ts / f"{cam_name}.jpg"
        if not img_path.is_file() or cam_name not in sensors:
            continue
        img_bgr = cv2.cvtColor(np.array(rrv.Image.open(img_path).convert("RGB")),
                               cv2.COLOR_RGB2BGR)
        if tag == "BEFORE":
            overlays = [(xyz_kept, RED_BGR, 3), (pts_removed, BLUE_BGR, 2)]
        else:
            overlays = [(xyz_kept, RED_BGR, 3)]
        panels.append(cam_panel(img_bgr, overlays, sensors[cam_name],
                                pose_samples, int(ts), cam_name))
    return panels


def compose(panels: list, title: str) -> np.ndarray:
    while len(panels) % 4:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    body = np.vstack(rows)
    header = np.zeros((44, body.shape[1], 3), np.uint8)
    cv2.putText(header, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (240, 240, 240), 2)
    return np.vstack([header, body])


def run_clip(clip: str) -> Path | None:
    pkl = MID / f"{clip}_cones_denoised.pkl"
    if not pkl.is_file():
        print(f"missing {pkl}")
        return None
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    fr = pick_best_frame(data)
    ts = fr["ts"]
    clip_dir = BACKUP / clip
    all_ts = rrv.list_clip_frames(clip_dir)
    pose_samples = rrv.build_clip_pose_samples(clip_dir, all_ts)

    before = frame_panels(fr, clip_dir, ts, pose_samples, "BEFORE")
    after = frame_panels(fr, clip_dir, ts, pose_samples, "AFTER")
    full = np.vstack([
        compose(before, f"BEFORE denoise  {clip[9:17]}  ts={ts}  "
                        f"(red=kept, blue=removed)"),
        compose(after, f"AFTER denoise  kept={len(fr.get('cone_points_xyz_denoised', []))}"),
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / f"denoise_compare_{clip[9:17]}.png"
    cv2.imwrite(str(out_png), full)
    print(out_png)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    args = ap.parse_args()
    clips = [args.clip] if args.clip else GOOD_CLIPS
    for c in clips:
        run_clip(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
