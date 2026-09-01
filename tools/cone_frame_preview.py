#!/usr/bin/env python3
"""单帧可视化预览：BEV + 7 路相机投影。

投影完全复用 render_robotruck_clip_video.py 的现有管线：
  parse_camera / camera_time_compensated_T / project_points / draw_projection
不自己实现任何投影数学。

用法:
  .venv_smoke/bin/python tools/cone_frame_preview.py --clip <clip_id> [--ts <ts>] [--out-dir exp/robotruck/cone_png]
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

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "exp/robotruck/raw_volume_cache"
MID = ROOT / "exp/robotruck/cone_mid"
OUT = ROOT / "exp/robotruck/cone_png"
CONE_ID = 10
PANEL_W, PANEL_H = 800, 450


def load_render_module():
    path = ROOT / "tools" / "render_robotruck_clip_video.py"
    spec = importlib.util.spec_from_file_location("rrv_preview", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rrv = load_render_module()


def pick_best_frame(clip: str) -> tuple[str, int]:
    pkl = MID / f"{clip}_cones.pkl"
    best_ts, best_n = None, -1
    if pkl.is_file():
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        for fr in d["per_frame"]:
            n = fr.get("cone_count_full_cloud", 0)
            if n > best_n:
                best_ts, best_n = fr["ts"], n
    if best_ts is None:
        frs = sorted((BACKUP / clip / "frames").iterdir())
        best_ts = frs[len(frs) // 2].name
    return best_ts, max(best_n, 0)


def bev_panel(xyz, cone_mask, ts: str, clip: str) -> np.ndarray:
    fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    other = xyz[~cone_mask]
    cone = xyz[cone_mask]
    ax.scatter(other[::5, 0], other[::5, 1], s=0.6, c="#9ecae1", linewidths=0)
    if len(cone):
        ax.scatter(cone[:, 0], cone[:, 1], s=5, c="orange", linewidths=0)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-15, 90)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.set_title(f"BEV  {clip[9:17]}  ts={ts}\norange=cone({int(cone_mask.sum())})",
                 fontsize=8, color="white")
    fig.patch.set_facecolor("black")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def cam_panel(img_bgr, xyz, pred, cone_mask, cam_doc, pose_samples, lidar_ts,
              cam_name: str, frame_idx: int) -> np.ndarray:
    """单相机投影 tile —— 与视频渲染路径一致的投影逻辑。"""
    K, dist5, T_c_v, cal_w, cal_h = rrv.parse_camera(cam_doc)
    cam_ts = int(cam_doc.get("timestamp") or int(lidar_ts))
    T_c_v_proj = rrv.camera_time_compensated_T(T_c_v, pose_samples, lidar_ts, cam_ts)

    ih, iw = img_bgr.shape[:2]
    if iw != cal_w or ih != cal_h:
        K = K.copy()
        K[0, :] *= iw / float(cal_w)
        K[1, :] *= ih / float(cal_h)

    # 全量点按语义类别着色（与视频一致）
    uv, cols = rrv.project_points(
        xyz, pred.astype(np.int32), K, dist5, T_c_v_proj, iw, ih,
        max_points=200000, seed=frame_idx,
    )
    proj = rrv.draw_projection(img_bgr, uv, cols, radius=2, alpha=0.85)

    # 锥桶点红色高亮（第二遍，画在最上层）
    if cone_mask.any():
        uv_c, _ = rrv.project_points(
            xyz[cone_mask], np.full(int(cone_mask.sum()), CONE_ID, np.int32),
            K, dist5, T_c_v_proj, iw, ih, max_points=200000, seed=frame_idx,
        )
        red = np.zeros((uv_c.shape[0], 3), np.uint8)
        red[:, 2] = 255
        proj = rrv.draw_projection(proj, uv_c, red, radius=4, alpha=1.0)

    vis = cv2.resize(proj, (PANEL_W, PANEL_H))
    cv2.putText(vis, f"{cam_name} cone_pts={int(cone_mask.sum())}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--ts", default="")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    clip = args.clip
    ts = args.ts or pick_best_frame(clip)[0]
    clip_dir = BACKUP / clip
    fr = clip_dir / "frames" / ts

    arr = np.fromfile(fr / "lidar_merge.bin", dtype=np.float32).reshape(-1, 7)
    xyz = arr[:, :3]
    pred = np.load(clip_dir / "preds" / f"{ts}_pred.npy").astype(np.int32).reshape(-1)
    cone_mask = pred == CONE_ID

    meta = rrv.json.loads((fr / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]
    all_ts = rrv.list_clip_frames(clip_dir)
    pose_samples = rrv.build_clip_pose_samples(clip_dir, all_ts)
    frame_idx = all_ts.index(ts) if ts in all_ts else 0

    panels = [bev_panel(xyz, cone_mask, ts, clip)]
    for cam_name in rrv.CAM_ORDER:
        img_path = fr / f"{cam_name}.jpg"
        if not img_path.is_file() or cam_name not in sensors:
            continue
        img = cv2.cvtColor(np.array(rrv.Image.open(img_path).convert("RGB")), cv2.COLOR_BGR2RGB)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        panel = cam_panel(img_bgr, xyz, pred, cone_mask, sensors[cam_name],
                          pose_samples, int(ts), cam_name, frame_idx)
        panels.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

    while len(panels) % 4:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    full = np.vstack(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"preview_{clip[9:17]}_{ts}.png"
    cv2.imwrite(str(out_png), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    print(out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
