#!/usr/bin/env python3
"""单帧可视化预览：BEV + 7 路相机投影。

投影完全复用 render_robotruck_clip_video.py 的现有管线：
  parse_camera / camera_time_compensated_T / project_points / draw_projection
不自己实现任何投影数学。

锥桶高亮只用 cluster_cones 后的点（丢弃墙/路沿假阳性 orphan），
并可选叠加质心十字。远距点 (> CONE_PROJ_Y_MAX) 只在 BEV 显示，避免投到地平线墙面。

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
# camera highlight: skip far cone clusters (geometrically land on horizon/wall)
CONE_PROJ_Y_MAX = 80.0
CONE_PROJ_Y_MIN = -40.0
# per-cluster point budget for overlay (avoid fat red clouds that look “shifted”)
CONE_PTS_PER_CLUSTER = 24
# RGB gate: keep cluster if any cam in CAM_ORDER finds orange near projected centroid
ORANGE_SEARCH = 16
ORANGE_MIN_FRAC = 0.02
RGB_GATE_CAMS = ("camera1", "camera2", "camera5", "camera6")


def load_render_module():
    path = ROOT / "tools" / "render_robotruck_clip_video.py"
    spec = importlib.util.spec_from_file_location("rrv_preview", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_denoise_module():
    path = ROOT / "tools" / "cone_filter_denoise.py"
    spec = importlib.util.spec_from_file_location("cfd_preview", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rrv = load_render_module()
cfd = load_denoise_module()


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


def cone_clusters_for_frame(clip: str, ts: str, cone_xyz: np.ndarray) -> list[dict]:
    """Prefer refreshed inline clustering (current thresholds); fall back to pkl."""
    clusters = cfd.cluster_cones(cone_xyz)
    if clusters:
        return clusters
    pkl = MID / f"{clip}_cones.pkl"
    if not pkl.is_file():
        return []
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    for fr in d.get("per_frame", []):
        if fr.get("ts") == ts:
            return list(fr.get("clusters") or [])
    return []


def masks_from_clusters(
    n_cloud: int,
    cone_idx: np.ndarray,
    clusters: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (clustered_mask, orphan_mask_on_cloud, centroids Nx3).

    clustered_mask / orphan are length n_cloud (full lidar).
    """
    clustered = np.zeros(n_cloud, dtype=bool)
    in_cone_kept = np.zeros(len(cone_idx), dtype=bool)
    cents = []
    for c in clusters:
        pids = np.asarray(c["point_ids"], dtype=np.int32)
        in_cone_kept[pids] = True
        clustered[cone_idx[pids]] = True
        cents.append(np.asarray(c["centroid_xyz"], dtype=np.float32))
    orphan = np.zeros(n_cloud, dtype=bool)
    if len(cone_idx):
        orphan[cone_idx[~in_cone_kept]] = True
    cents_arr = np.stack(cents, axis=0) if cents else np.zeros((0, 3), np.float32)
    return clustered, orphan, cents_arr


def _orange_frac(img_rgb: np.ndarray, u: int, v: int, rad: int = ORANGE_SEARCH) -> float:
    h, w = img_rgb.shape[:2]
    y0, y1 = max(0, v - rad), min(h, v + rad + 1)
    x0, x1 = max(0, u - rad), min(w, u + rad + 1)
    patch = img_rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    r = patch[..., 0].astype(np.float32)
    g = patch[..., 1].astype(np.float32)
    b = patch[..., 2].astype(np.float32)
    return float(((r > 100) & (r > g) & (g > b) & ((r - b) > 30)).mean())


def filter_clusters_by_rgb(
    clusters: list[dict],
    sensors: dict,
    pose_samples,
    lidar_ts: int,
    frame_dir: Path,
) -> list[dict]:
    """Drop clusters whose centroid never lands near orange in any gated camera.

    Projection math itself is unchanged — this only rejects semseg false positives
    (barrier / curb / chevron) that otherwise look like “wrong projection”.
    """
    if not clusters:
        return clusters
    kept = []
    for c in clusters:
        cent = np.asarray(c["centroid_xyz"], dtype=np.float64)
        if not (CONE_PROJ_Y_MIN <= float(cent[1]) <= CONE_PROJ_Y_MAX):
            # out of highlight band: keep for BEV, skip RGB gate
            kept.append(c)
            continue
        ok = False
        saw_cam = False
        for cam_name in RGB_GATE_CAMS:
            img_path = frame_dir / f"{cam_name}.jpg"
            if cam_name not in sensors or not img_path.is_file():
                continue
            cam_doc = sensors[cam_name]
            K, dist5, T_c_v, cal_w, cal_h = rrv.parse_camera(cam_doc)
            img = np.array(rrv.Image.open(img_path).convert("RGB"))
            ih, iw = img.shape[:2]
            if iw != cal_w or ih != cal_h:
                K = K.copy()
                K[0, :] *= iw / float(cal_w)
                K[1, :] *= ih / float(cal_h)
            T = rrv.camera_time_compensated_T(
                T_c_v, pose_samples, lidar_ts, int(cam_doc.get("timestamp") or lidar_ts),
            )
            pc = (T @ np.array([cent[0], cent[1], cent[2], 1.0]))[:3]
            if pc[2] < 0.3:
                continue
            uv, _ = cv2.projectPoints(
                pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist5,
            )
            u, v = uv.reshape(2)
            if not (0 <= u < iw and 0 <= v < ih):
                continue
            saw_cam = True
            if _orange_frac(img, int(u), int(v)) >= ORANGE_MIN_FRAC:
                ok = True
                break
        if ok or not saw_cam:
            # keep if orange-confirmed, or if no gated cam saw the centroid (don't over-kill)
            kept.append(c)
    return kept


def sparse_cluster_points(
    xyz: np.ndarray,
    cone_idx: np.ndarray,
    clusters: list[dict],
) -> np.ndarray:
    """Boolean mask (full cloud) with capped points per cluster for clean overlay."""
    mask = np.zeros(len(xyz), dtype=bool)
    rng = np.random.default_rng(0)
    for c in clusters:
        pids = np.asarray(c["point_ids"], dtype=np.int32)
        if pids.size > CONE_PTS_PER_CLUSTER:
            pids = rng.choice(pids, size=CONE_PTS_PER_CLUSTER, replace=False)
        mask[cone_idx[pids]] = True
    return mask


def bev_panel(xyz, clustered_mask, orphan_mask, ts: str, clip: str) -> np.ndarray:
    fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    other = xyz[~(clustered_mask | orphan_mask)]
    ax.scatter(other[::5, 0], other[::5, 1], s=0.6, c="#9ecae1", linewidths=0)
    if orphan_mask.any():
        o = xyz[orphan_mask]
        ax.scatter(o[:, 0], o[:, 1], s=2, c="#555555", linewidths=0, alpha=0.5)
    if clustered_mask.any():
        c = xyz[clustered_mask]
        ax.scatter(c[:, 0], c[:, 1], s=6, c="orange", linewidths=0)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-15, 90)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.set_title(
        f"BEV  {clip[9:17]}  ts={ts}\n"
        f"orange=clustered({int(clustered_mask.sum())})  gray=orphanFP({int(orphan_mask.sum())})",
        fontsize=8, color="white",
    )
    fig.patch.set_facecolor("black")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def cam_panel(
    img_bgr, xyz, pred, highlight_mask, orphan_mask, centroids, cam_doc, pose_samples,
    lidar_ts, cam_name: str, frame_idx: int,
) -> np.ndarray:
    """单相机投影 tile —— 与视频渲染路径一致的投影逻辑；锥桶只画稀疏聚类点+质心。"""
    K, dist5, T_c_v, cal_w, cal_h = rrv.parse_camera(cam_doc)
    cam_ts = int(cam_doc.get("timestamp") or int(lidar_ts))
    T_c_v_proj = rrv.camera_time_compensated_T(T_c_v, pose_samples, lidar_ts, cam_ts)

    ih, iw = img_bgr.shape[:2]
    if iw != cal_w or ih != cal_h:
        K = K.copy()
        K[0, :] *= iw / float(cal_w)
        K[1, :] *= ih / float(cal_h)

    # 语义底图：剥掉全部 class10（含 orphan），避免墙/路沿假阳性被画成橙色“错投影”
    pred_vis = pred.astype(np.int32).copy()
    pred_vis[pred_vis == CONE_ID] = -1
    uv, cols = rrv.project_points(
        xyz, pred_vis, K, dist5, T_c_v_proj, iw, ih,
        max_points=200000, seed=frame_idx,
    )
    proj = rrv.draw_projection(img_bgr, uv, cols, radius=2, alpha=0.40)

    # 锥桶：稀疏聚类点 + 合理纵向范围
    hi = highlight_mask & (xyz[:, 1] >= CONE_PROJ_Y_MIN) & (xyz[:, 1] <= CONE_PROJ_Y_MAX)
    n_hi = int(hi.sum())
    if n_hi:
        uv_c, _ = rrv.project_points(
            xyz[hi], np.full(n_hi, CONE_ID, np.int32),
            K, dist5, T_c_v_proj, iw, ih, max_points=200000, seed=frame_idx,
        )
        red = np.zeros((uv_c.shape[0], 3), np.uint8)
        red[:, 2] = 255
        proj = rrv.draw_projection(proj, uv_c, red, radius=3, alpha=1.0)

    n_cents = 0
    if len(centroids):
        cm = (centroids[:, 1] >= CONE_PROJ_Y_MIN) & (centroids[:, 1] <= CONE_PROJ_Y_MAX)
        cents = centroids[cm]
        n_cents = int(len(cents))
        # vertical stick: mid-xy at cluster min_z / max_z so marker covers cone body height
        # (centroid alone sits near ground lidar returns and looks “low”)
        sticks = []
        for c_xyz in cents:
            # approximate height from nearby highlight points
            dxy = np.linalg.norm(xyz[highlight_mask][:, :2] - c_xyz[:2], axis=1) if highlight_mask.any() else np.array([])
            if len(dxy) and (dxy < 0.6).any():
                zs = xyz[highlight_mask][dxy < 0.6, 2]
                z0, z1 = float(zs.min()), float(zs.max())
            else:
                z0, z1 = float(c_xyz[2]) - 0.15, float(c_xyz[2]) + 0.55
            sticks.append([c_xyz[0], c_xyz[1], z0])
            sticks.append([c_xyz[0], c_xyz[1], z1])
        if sticks:
            # project pairwise for vertical sticks (avoid depth-sort scrambling pairs)
            ones = np.ones((len(sticks), 1), np.float64)
            ph = np.hstack([np.asarray(sticks, np.float64), ones])
            pc = (T_c_v_proj @ ph.T).T[:, :3]
            for i in range(0, len(sticks), 2):
                if pc[i, 2] < 0.3 or pc[i + 1, 2] < 0.3:
                    continue
                uv, _ = cv2.projectPoints(
                    pc[i:i + 2].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist5,
                )
                (u0, v0), (u1, v1) = uv.reshape(-1, 2)
                if not np.isfinite([u0, v0, u1, v1]).all():
                    continue
                cv2.line(proj, (int(u0), int(v0)), (int(u1), int(v1)), (255, 0, 255), 2)
                cv2.drawMarker(
                    proj, (int(0.5 * (u0 + u1)), int(0.5 * (v0 + v1))), (255, 0, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2,
                )

    vis = cv2.resize(proj, (PANEL_W, PANEL_H))
    cv2.putText(
        vis,
        f"{cam_name} pts={n_hi} cents={n_cents} orphan={int(orphan_mask.sum())}",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--ts", default="")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--no-rgb-gate", action="store_true",
                    help="Skip orange-pixel gate on cluster centroids")
    args = ap.parse_args()

    clip = args.clip
    ts = args.ts or pick_best_frame(clip)[0]
    clip_dir = BACKUP / clip
    fr = clip_dir / "frames" / ts

    arr = np.fromfile(fr / "lidar_merge.bin", dtype=np.float32).reshape(-1, 7)
    xyz = arr[:, :3]
    pred = np.load(clip_dir / "preds" / f"{ts}_pred.npy").astype(np.int32).reshape(-1)
    cone_mask = pred == CONE_ID
    cone_idx = np.where(cone_mask)[0]
    clusters = cone_clusters_for_frame(clip, ts, xyz[cone_mask])

    meta = rrv.json.loads((fr / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]
    all_ts = rrv.list_clip_frames(clip_dir)
    pose_samples = rrv.build_clip_pose_samples(clip_dir, all_ts)
    frame_idx = all_ts.index(ts) if ts in all_ts else 0

    n_before = len(clusters)
    if not args.no_rgb_gate:
        clusters = filter_clusters_by_rgb(
            clusters, sensors, pose_samples, int(ts), fr,
        )
    clustered_mask, orphan_mask, centroids = masks_from_clusters(
        len(xyz), cone_idx, clusters,
    )
    # orphans = raw class10 minus ANY spatial cluster before RGB gate would differ;
    # recompute orphan vs raw for honest BEV: everything class10 not in kept clusters
    highlight_mask = sparse_cluster_points(xyz, cone_idx, clusters)

    panels = [bev_panel(xyz, clustered_mask, orphan_mask, ts, clip)]
    for cam_name in rrv.CAM_ORDER:
        img_path = fr / f"{cam_name}.jpg"
        if not img_path.is_file() or cam_name not in sensors:
            continue
        img_bgr = cv2.cvtColor(
            np.array(rrv.Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR,
        )
        panel = cam_panel(
            img_bgr, xyz, pred, highlight_mask, orphan_mask, centroids, sensors[cam_name],
            pose_samples, int(ts), cam_name, frame_idx,
        )
        panels.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

    while len(panels) % 4:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    full = np.vstack(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"preview_{clip[9:17]}_{ts}.png"
    cv2.imwrite(str(out_png), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    print(
        f"{out_png}  raw_cone={int(cone_mask.sum())} "
        f"clusters={n_before}->{len(clusters)} "
        f"highlight_pts={int(highlight_mask.sum())} orphan={int(orphan_mask.sum())}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
