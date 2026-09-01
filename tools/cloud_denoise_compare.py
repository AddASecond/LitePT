#!/usr/bin/env python3
"""全点云纯密度去噪对比 PNG（无类别、无标签，仅 xyz）。

BEFORE: 全点云按高度着色 + 被剔除点洋红色高亮
AFTER : 去噪后保留点按高度着色
布局同已确认样式（BEV + 7 相机），投影复用 rrv 管线。

用法:
  .venv_smoke/bin/python tools/cloud_denoise_compare.py            # 全部 7 case
  .venv_smoke/bin/python tools/cloud_denoise_compare.py --clip <clip_id>
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
BACKUP = ROOT / "exp/robotruck/raw_volume_cache"
OUT = ROOT / "exp/robotruck/cone_png"
PANEL_W, PANEL_H = 800, 450

from density_denoise import denoise_cloud  # noqa: E402

GOOD_CLIPS = [
    "batch_s5_11c2aa2d-2618-45d8-ab28-5cf1529eca84",
    "batch_s5_6bc60101-e684-438b-9983-b74c1cc5fe2b",
    "batch_s5_f1683bae-a2a6-496a-b9a7-9cc2ccfbc3e0",
    "batch_s5_72ebcd30-693f-417e-855d-ae66447d2020",
    "batch_s5_9d17f68b-3594-4412-b39e-deb7e7250de9",
    "batch_s5_701ff460-2b29-4433-b492-39e7ae2310b6",
    "batch_s5_c5d9f4b4-e410-4241-8e46-f5653db90f5b",
]

MAGENTA_BGR = (255, 0, 255)


def load_render_module():
    path = ROOT / "tools" / "render_robotruck_clip_video.py"
    spec = importlib.util.spec_from_file_location("rrv_cloudcmp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rrv = load_render_module()


def pick_densest_frame(clip_dir: Path) -> str:
    """按 lidar_merge.bin 文件大小选点数最多的帧（N = size/28）。"""
    best_ts, best_n = "", -1
    for p in sorted((clip_dir / "frames").glob("*/lidar_merge.bin")):
        n = p.stat().st_size // 28
        if n > best_n:
            best_ts, best_n = p.parent.name, n
    return best_ts


def z_to_bgr(z: np.ndarray, zmin: float = -2.0, zmax: float = 6.0) -> np.ndarray:
    """高度着色：低=蓝，中=绿，高=红。返回 uint8 (N,3) BGR。"""
    t = np.clip((z - zmin) / (zmax - zmin), 0, 1)
    b = np.clip(1.5 - 3 * np.abs(t - 0.0), 0, 1)
    g = np.clip(1.5 - 3 * np.abs(t - 0.5), 0, 1)
    r = np.clip(1.5 - 3 * np.abs(t - 1.0), 0, 1)
    cols = np.stack([b, g, r], axis=1)
    return (cols * 255).astype(np.uint8)


def bev_panel(xyz, removed, ts: str, tag: str, n_kept: int) -> np.ndarray:
    fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    step = max(1, len(xyz) // 60000)
    cols = z_to_bgr(xyz[:, 2])[::step][:, ::-1].astype(float) / 255.0  # BGR→RGB 0-1
    ax.scatter(xyz[::step, 0], xyz[::step, 1], s=0.5, c=cols, linewidths=0)
    if removed is not None and len(removed):
        ax.scatter(removed[::3, 0], removed[::3, 1], s=2, c="magenta", linewidths=0)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-15, 90)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("white")
    extra = f" magenta=removed({len(removed)})" if removed is not None else ""
    ax.set_title(f"{tag} BEV  ts={ts}\nall={len(xyz)} kept={n_kept}{extra}",
                 fontsize=8, color="white")
    fig.patch.set_facecolor("black")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def cam_panel(img_bgr, xyz, cols_bgr, overlays, cam_doc, pose_samples,
              lidar_ts, cam_name: str) -> np.ndarray:
    K, dist5, T_c_v, cal_w, cal_h = rrv.parse_camera(cam_doc)
    cam_ts = int(cam_doc.get("timestamp") or int(lidar_ts))
    T_c_v_proj = rrv.camera_time_compensated_T(T_c_v, pose_samples, lidar_ts, cam_ts)
    ih, iw = img_bgr.shape[:2]
    if iw != cal_w or ih != cal_h:
        K = K.copy()
        K[0, :] *= iw / float(cal_w)
        K[1, :] *= ih / float(cal_h)

    vis = img_bgr.copy()
    uv, _ = rrv.project_points(
        xyz, np.zeros(len(xyz), np.int32), K, dist5, T_c_v_proj, iw, ih,
        max_points=250000, seed=0,
    )
    # project_points 可能下采样点数，颜色按其顺序未知 → 逐色叠加成本高；
    # 这里统一用点自身颜色: project_points 返回 uv 数与输入一一对应（未下采样时）
    cols = cols_bgr[:uv.shape[0]] if uv.shape[0] <= len(xyz) else cols_bgr
    vis = rrv.draw_projection(vis, uv, cols, radius=1, alpha=0.9)
    for pts, color, radius in overlays:
        if len(pts) == 0:
            continue
        u2, _ = rrv.project_points(
            pts, np.zeros(len(pts), np.int32), K, dist5, T_c_v_proj, iw, ih,
            max_points=400000, seed=0,
        )
        c2 = np.zeros((u2.shape[0], 3), np.uint8)
        c2[:] = color
        vis = rrv.draw_projection(vis, u2, c2, radius=radius, alpha=1.0)

    vis = cv2.resize(vis, (PANEL_W, PANEL_H))
    cv2.putText(vis, f"{cam_name} pts={len(xyz)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def run_clip(clip: str) -> Path | None:
    clip_dir = BACKUP / clip
    ts = pick_densest_frame(clip_dir)
    bin_path = clip_dir / "frames" / ts / "lidar_merge.bin"
    arr = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 7)
    xyz = arr[:, :3].astype(np.float64)

    kept, info = denoise_cloud(xyz)
    xyz_kept = xyz[kept]
    xyz_removed = xyz[~kept]
    print(f"[{clip[9:17]}] ts={ts} in={info['n_in']} kept={info['n_kept']} "
          f"noise={info['n_noise']} components={info['n_components']}", flush=True)

    all_ts = rrv.list_clip_frames(clip_dir)
    pose_samples = rrv.build_clip_pose_samples(clip_dir, all_ts)
    meta = rrv.json.loads((clip_dir / "frames" / ts / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]

    def panels_for(xyz_show, removed_show, tag):
        cols = z_to_bgr(xyz_show[:, 2].astype(np.float64))
        ps = [cv2.cvtColor(
            bev_panel(xyz_show, removed_show, ts, tag, len(xyz_kept)),
            cv2.COLOR_RGB2BGR)]
        overlays = [(xyz_removed, MAGENTA_BGR, 2)] if removed_show is not None else []
        for cam_name in rrv.CAM_ORDER:
            img_path = clip_dir / "frames" / ts / f"{cam_name}.jpg"
            if not img_path.is_file() or cam_name not in sensors:
                continue
            img_bgr = cv2.cvtColor(
                np.array(rrv.Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
            ps.append(cam_panel(img_bgr, xyz_show, cols, overlays,
                                sensors[cam_name], pose_samples, int(ts), cam_name))
        return ps

    def compose(panels, title):
        while len(panels) % 4:
            panels.append(np.zeros_like(panels[0]))
        rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
        body = np.vstack(rows)
        header = np.zeros((44, body.shape[1], 3), np.uint8)
        cv2.putText(header, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (240, 240, 240), 2)
        return np.vstack([header, body])

    full = np.vstack([
        compose(panels_for(xyz, xyz_removed, "BEFORE"),
                f"BEFORE pure-density denoise  {clip[9:17]}  ts={ts}  "
                f"(height-colored, magenta=removed)"),
        compose(panels_for(xyz_kept, None, "AFTER"),
                f"AFTER  kept={info['n_kept']}/{info['n_in']}  "
                f"(SOR k=16 α=1.0, eps=0.25m, min_component=5)"),
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / f"cloud_denoise_compare_{clip[9:17]}.png"
    cv2.imwrite(str(out_png), full)
    print(str(out_png), flush=True)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    args = ap.parse_args()
    for c in ([args.clip] if args.clip else GOOD_CLIPS):
        run_clip(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
