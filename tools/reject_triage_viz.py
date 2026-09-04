#!/usr/bin/env python3
"""REJECT triage 可视化：BEV（框出可疑区）+ 多相机投影。

解决「斜视图看不出问题 / 文字对不上图」：
  1. BEV 聚合（帧序着色 → 鬼影显彩虹分层）
  2. 红色方框标出轨迹/点云可疑区域 + 锚点十字
  3. 锚点帧 lidar→相机投影（错位/分层在图像上更直观）
  4. 完整 bag_name + clip_id + 中英文说明

输出: exp/robotruck/reject_triage_viz/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

LIST = ROOT / "exp/robotruck/pose_badcase/final_badcase_list.json"
METRICS = ROOT / "exp/robotruck/pose_badcase/v2_metrics.json"
CACHE = ROOT / "exp/robotruck/raw_volume_cache"
OUTDIR = ROOT / "exp/robotruck/reject_triage_viz"

MONGO_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)
DB = "perception_experiment"
CLIPS_COL = "raw_data_clips_lidar14_0813"
FRAMES_COL = "raw_data_frames_lidar14_0813"
RAW_ROOTS = [Path(f"/data/rawdata{s}") for s in ("", "-1", "-2", "-3", "-4")]

N_AGG = 30
BEV_MARGIN_M = 25.0
BEV_FULL_PAD_M = 40.0
SUB_PER_FRAME = 15000
CAM_PANEL_W, CAM_PANEL_H = 640, 360
CAM_ORDER = ["camera1", "camera2", "camera5", "camera6", "camera7", "camera8"]
HEADER_H = 220


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ls = _load("layer_scan", ROOT / "tools/layer_scan.py")
vrp = _load("vrp", ROOT / "tools/validate_raw_single_frame_projection.py")


def parse_camera(cam_doc: dict):
    """Same as render_robotruck_clip_video.parse_camera (avoid loading that CUDA module)."""
    K = np.asarray(cam_doc["intrinsic"]["intrinsic"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(cam_doc["intrinsic"]["distortion"], dtype=np.float64).reshape(-1)
    dist5 = np.zeros(5, dtype=np.float64)
    dist5[: min(5, dist.size)] = dist[:5]
    T_v_c = np.asarray(cam_doc["extrinsic"]["transformation"], dtype=np.float64).reshape(4, 4)
    T_c_v = np.linalg.inv(T_v_c)
    w = int(cam_doc["intrinsic"]["width"])
    h = int(cam_doc["intrinsic"]["height"])
    return K, dist5, T_c_v, w, h


def load_rejects() -> list[dict]:
    return [r for r in json.loads(LIST.read_text()) if r.get("tier") == "REJECT"]


def load_metrics() -> dict[str, dict]:
    if not METRICS.is_file():
        return {}
    return {m["clip_id"]: m for m in json.loads(METRICS.read_text())}


def fetch_bag_meta(clip_ids: list[str]) -> dict[str, dict]:
    from pymongo import MongoClient

    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    out = {}
    for doc in c[DB][CLIPS_COL].find(
        {"clip_id": {"$in": clip_ids}}, {"clip_id": 1, "bag_name": 1}
    ):
        out[doc["clip_id"]] = {"bag_name": doc.get("bag_name") or ""}
    return out


def explain_en(flags: list[str], row: dict) -> list[str]:
    if any(f.startswith("GATE:high_confidence_pose_drift") for f in flags):
        return [
            "Why: kinematics OK but pose-shift search prefers non-zero displacement (systematic drift).",
            "BEV red box: look for rainbow layering / double lines on guardrail & curb.",
            "Cameras: check poles/walls for misalignment vs image.",
        ]
    if any(f.startswith("GATE:layering_with_pose_inconsistency") for f in flags):
        return [
            "Why: lidar height-layering AND pose misalignment both fire.",
            "BEV red box: stacked/rainbow sheets of the same structure.",
            "Cameras: cloud vs road/guardrail offset.",
        ]
    if any(f.startswith("PC_EXTREME") for f in flags):
        return [
            f"Why: extreme cross-frame mismatch at jump window (anom_pc={row.get('max_anom_pc')}m).",
            "BEV red box: duplicated structures (ghosting) near ANCHOR.",
            "Cameras: context for the anchor time; focus on BEV box.",
        ]
    if any(f.startswith("PC_CONFIRMED") for f in flags):
        return [
            f"Why: local mismatch + global pc_p20 elevated (gpc={row.get('global_pc_p20')}m).",
            "BEV red box: confirmed layering; outside should look cleaner.",
            "Cameras: compare static structure projection consistency.",
        ]
    return [f"Why: {','.join(flags)}", "BEV red box = suspect; cameras = anchor-frame projection."]


def explain_zh(flags: list[str], row: dict) -> list[str]:
    if any(f.startswith("GATE:high_confidence_pose_drift") for f in flags):
        return [
            "判定: 运动学可达，但 pose 平移搜索一致偏向非零位移 → 系统性漂移。",
            "BEV红框: 漂移最明显的聚合窗口；看框内护栏/路缘是否彩虹分层或双线。",
            "相机投影: 看静态结构（杆/护栏）是否相对图像错位或重影。",
        ]
    if any(f.startswith("GATE:layering_with_pose_inconsistency") for f in flags):
        return [
            "判定: 雷达高度分层 + pose 对齐失败同时成立。",
            "BEV红框: 问题锚点邻域；框内应看到多层/彩虹鬼影。",
            "相机投影: 点云贴路面/护栏是否整体偏移。",
        ]
    if any(f.startswith("PC_EXTREME") for f in flags):
        return [
            f"判定: 诡异窗口跨帧对齐误差极大 (anom_pc={row.get('max_anom_pc')}m)。",
            "BEV红框: 轨迹跳变/不可达锚点邻域；框内同一结构被复制多份。",
            "相机投影: 辅助确认该时刻场景；重点看 BEV 框内鬼影。",
        ]
    if any(f.startswith("PC_CONFIRMED") for f in flags):
        return [
            f"判定: 局部错位 + 全局 pc_p20 抬升 (gpc={row.get('global_pc_p20')}m)。",
            "BEV红框: 确认区分层区域；框外相对更干净。",
            "相机投影: 对照红框内结构在图像上的投影一致性。",
        ]
    if any(f.startswith("PC_BORDERLINE") for f in flags):
        return [
            f"判定: 局部 PC 异常偏高但未过 REJECT 确认门 (anom={row.get('max_anom_pc')} "
            f"med={row.get('med_anom_pc')} gpc={row.get('global_pc_p20')})。",
            "BEV红框: 重点看是否真分层/鬼影，还是天桥/单尖刺虚高。",
            "相机投影: 对照锚点附近静态结构是否错位。",
        ]
    if any(f.startswith("POSE_SHIFT_WEAK") for f in flags):
        return [
            f"判定: pose-shift 偏向非零位移但证据不够强 (shift={row.get('med_shift')}m)。",
            "BEV红框: 看护栏/路缘是否轻微彩虹分层或双线。",
            "相机投影: 杆/墙相对图像是否系统性偏移。",
        ]
    if any(f.startswith("INFEASIBLE_ONLY") for f in flags):
        return [
            f"判定: 仅运动学不可达 (n_infeas={row.get('n_infeas')})，PC/pose 未确认分层。",
            "BEV红框: 轨迹跳变邻域；多数可能是瞬移尖刺而非真实分层。",
            "相机投影: 辅助看场景；若框内干净可忽略。",
        ]
    return ["判定: " + ",".join(flags), "BEV红框标出可疑区；相机为锚点帧投影。"]


def pick_anchor(frames: list, metrics: dict | None) -> tuple[int, str]:
    valid_i = [i for i, f in enumerate(frames) if f[1] is not None]
    if metrics and metrics.get("anomaly_centers") and valid_i:
        c0 = int(metrics["anomaly_centers"][0])
        if 0 <= c0 < len(valid_i):
            return int(valid_i[c0]), "anomaly_center"
    # fallback worst |d2|
    if len(valid_i) < 3:
        return (valid_i[len(valid_i) // 2] if valid_i else 0), "mid"
    P = np.array([frames[i][1] for i in valid_i], float)
    d2 = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=1)
    return int(valid_i[int(np.argmax(d2)) + 1]), "worst_d2"


def neighbor_idxs(n: int, wi: int, n_agg: int = N_AGG) -> list[int]:
    half = n_agg // 2
    lo = max(0, wi - half)
    hi = min(n, lo + n_agg)
    lo = max(0, hi - n_agg)
    return list(range(lo, hi))


def cache_dir(clip_id: str) -> Path | None:
    for p in (CACHE / f"batch_s5_{clip_id}", CACHE / clip_id):
        if (p / "frames").is_dir():
            return p
    return None


def aggregate_map_clouds(
    clip_id: str, frames: list, idxs: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz_map Nx3, frame_color_idx N, traj Mx3."""
    chunks, fids, traj = [], [], []
    for k, i in enumerate(idxs):
        ts, t, R, md5 = frames[i]
        if t is None or R is None or not md5:
            continue
        try:
            xyz = ls.load_cloud(clip_id, str(ts), md5)
        except Exception:
            continue
        if len(xyz) > SUB_PER_FRAME:
            xyz = xyz[:: max(1, len(xyz) // SUB_PER_FRAME)]
        chunks.append(xyz @ R.T + t)
        fids.append(np.full(len(xyz), k, np.int32))
        traj.append(t)
    if not chunks:
        return np.zeros((0, 3)), np.zeros((0,), np.int32), np.zeros((0, 3))
    return np.concatenate(chunks), np.concatenate(fids), np.asarray(traj, float)


def suspect_box_xy(
    traj: np.ndarray, anchor_xy: np.ndarray, margin: float = BEV_MARGIN_M
) -> tuple[float, float, float, float]:
    """Axis-aligned box covering anomaly traj segment + margin."""
    if len(traj) == 0:
        x, y = float(anchor_xy[0]), float(anchor_xy[1])
        return x - margin, y - margin, x + margin, y + margin
    xs = traj[:, 0]
    ys = traj[:, 1]
    return (
        float(xs.min() - margin),
        float(ys.min() - margin),
        float(xs.max() + margin),
        float(ys.max() + margin),
    )


def render_bev(
    pts: np.ndarray,
    fids: np.ndarray,
    traj_all: np.ndarray,
    traj_win: np.ndarray,
    anchor_xy: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    zoom: bool,
    title: str,
    width: int = 1200,
    height: int = 1200,
) -> np.ndarray:
    """BEV image with trajectory, red suspect box, anchor cross."""
    img = np.full((height, width, 3), 20, np.uint8)
    if len(pts) == 0 and len(traj_all) == 0:
        cv2.putText(img, "no points", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return img

    x0, y0, x1, y1 = box
    if zoom:
        xmin, xmax, ymin, ymax = x0, x1, y0, y1
    else:
        if len(traj_all):
            lo = np.percentile(traj_all[:, :2], 2, axis=0)
            hi = np.percentile(traj_all[:, :2], 98, axis=0)
        else:
            lo = np.array([x0, y0])
            hi = np.array([x1, y1])
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + BEV_FULL_PAD_M
        half = max(half, 50.0)
        xmin, xmax = cx - half, cx + half
        ymin, ymax = cy - half, cy + half

    def to_px(xy):
        u = ((xy[..., 0] - xmin) / max(1e-6, xmax - xmin) * (width - 1)).astype(np.int32)
        # y up in map → image y down
        v = ((ymax - xy[..., 1]) / max(1e-6, ymax - ymin) * (height - 1)).astype(np.int32)
        return u, v

    # points colored by frame order
    if len(pts):
        keep = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
            & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        )
        p = pts[keep]
        f = fids[keep] if len(fids) == len(pts) else np.zeros(len(p), np.int32)
        if len(p):
            u, v = to_px(p[:, :2])
            ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            u, v, f = u[ok], v[ok], f[ok]
            if len(f):
                tnorm = f.astype(np.float32) / max(1, int(f.max()) if len(f) else 1)
                gray = (tnorm * 255).astype(np.uint8).reshape(-1, 1)
                cols = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO).reshape(-1, 3)
                img[v, u] = cols

    # full traj thin gray
    if len(traj_all) >= 2:
        u, v = to_px(traj_all[:, :2])
        for i in range(len(u) - 1):
            cv2.line(img, (int(u[i]), int(v[i])), (int(u[i + 1]), int(v[i + 1])),
                     (90, 90, 90), 1, cv2.LINE_AA)

    # window traj thick yellow
    if len(traj_win) >= 2:
        u, v = to_px(traj_win[:, :2])
        for i in range(len(u) - 1):
            cv2.line(img, (int(u[i]), int(v[i])), (int(u[i + 1]), int(v[i + 1])),
                     (0, 220, 255), 3, cv2.LINE_AA)

    # red suspect box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    uc, vc = to_px(corners)
    poly = np.stack([uc, vc], axis=1).astype(np.int32)
    cv2.polylines(img, [poly], True, (0, 0, 255), 4, cv2.LINE_AA)
    # label
    lu, lv = int(uc[0]), int(vc[0])
    cv2.putText(img, "SUSPECT", (lu + 6, max(24, lv - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, "SUSPECT", (lu + 6, max(24, lv - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # anchor cross
    au, av = to_px(anchor_xy.reshape(1, 2))
    au, av = int(au[0]), int(av[0])
    cv2.drawMarker(img, (au, av), (0, 255, 255), cv2.MARKER_CROSS, 36, 3, cv2.LINE_AA)
    cv2.putText(img, "ANCHOR", (au + 12, av - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img, title, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, title, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img, "color=frame order (rainbow = ghosting)", (16, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return img


def load_frame_doc(clip_id: str, ts: int) -> dict | None:
    from pymongo import MongoClient

    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return c[DB][FRAMES_COL].find_one({"clip_id": clip_id, "timestamp": int(ts)})


def load_camera_image(cam_doc: dict) -> np.ndarray | None:
    md5 = (cam_doc or {}).get("md5")
    if not md5 or len(md5) != 32:
        return None
    try:
        path = vrp.content_path(md5, "camera", ".jpg", RAW_ROOTS)
    except FileNotFoundError:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def project_cam_panel(
    xyz_veh: np.ndarray,
    cam_name: str,
    cam_doc: dict,
    image: np.ndarray,
) -> np.ndarray:
    """Project lidar (height-colored) onto camera image, resized panel."""
    panel = np.full((CAM_PANEL_H, CAM_PANEL_W, 3), 30, np.uint8)
    if image is None or cam_doc is None or len(xyz_veh) == 0:
        cv2.putText(panel, f"{cam_name}: no data", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return panel
    try:
        K, dist5, T_c_v, cal_w, cal_h = parse_camera(cam_doc)
    except Exception as exc:
        cv2.putText(panel, f"{cam_name}: calib err", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return panel
    ih, iw = image.shape[:2]
    K = K.copy()
    if iw != cal_w or ih != cal_h:
        K[0, :] *= iw / float(cal_w)
        K[1, :] *= ih / float(cal_h)
    # fake labels from height for color
    z = xyz_veh[:, 2]
    z0, z1 = np.percentile(z, 5), np.percentile(z, 95)
    tnorm = np.clip((z - z0) / max(1e-3, z1 - z0), 0, 1)
    # map to class ids 0..21 for WAYMO palette abuse — instead use project with custom colors
    ones = np.ones((len(xyz_veh), 1), float)
    ph = np.hstack([xyz_veh.astype(float), ones])
    pc = (T_c_v @ ph.T).T[:, :3]
    front = pc[:, 2] > 0.3
    pc = pc[front]
    tnorm = tnorm[front]
    if len(pc) == 0:
        out = image.copy()
    else:
        uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist5)
        uv = uv.reshape(-1, 2)
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < iw)
            & (uv[:, 1] >= 0) & (uv[:, 1] < ih)
            & np.isfinite(uv).all(axis=1)
        )
        uv = uv[inside]
        tnorm = tnorm[inside]
        zc = pc[inside][:, 2]
        order = np.argsort(-zc)
        uv, tnorm = uv[order], tnorm[order]
        gray = (tnorm * 255).astype(np.uint8).reshape(-1, 1)
        cols = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO).reshape(-1, 3)
        out = image.copy()
        overlay = out.copy()
        step = max(1, len(uv) // 80000)
        for (uu, vv), c in zip(uv[::step], cols[::step]):
            cv2.circle(overlay, (int(uu), int(vv)), 2, (int(c[0]), int(c[1]), int(c[2])), -1)
        out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)

    # resize
    panel = cv2.resize(out, (CAM_PANEL_W, CAM_PANEL_H), interpolation=cv2.INTER_AREA)
    cv2.putText(panel, cam_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(panel, cam_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return panel


def get_anchor_xyz_and_cams(
    clip_id: str, frames: list, wi: int
) -> tuple[np.ndarray, dict, dict]:
    """Vehicle-frame cloud + sensors doc + images for cams at anchor."""
    ts, t, R, md5 = frames[wi]
    xyz = ls.load_cloud(clip_id, str(ts), md5)
    # try cache frame.json for sensors
    sensors = {}
    cdir = cache_dir(clip_id)
    if cdir is not None:
        # nearest cached timestamp
        fr_dirs = sorted((cdir / "frames").iterdir(), key=lambda p: p.name)
        if fr_dirs:
            nearest = min(fr_dirs, key=lambda p: abs(int(p.name) - int(ts)))
            meta_p = nearest / "frame.json"
            if meta_p.is_file():
                meta = json.loads(meta_p.read_text())
                sensors = ((meta.get("dependency") or {}).get("sensors") or {})
                # if nearest != anchor, still use its images as approx; prefer exact
                exact = cdir / "frames" / str(ts)
                use = exact if exact.is_dir() else nearest
                imgs = {}
                for cam in CAM_ORDER:
                    jp = use / f"{cam}.jpg"
                    if jp.is_file():
                        imgs[cam] = cv2.imread(str(jp), cv2.IMREAD_COLOR)
                if imgs:
                    return xyz, sensors, imgs

    # mongo + rawdata
    doc = load_frame_doc(clip_id, int(ts))
    if not doc:
        return xyz, {}, {}
    sensors = ((doc.get("dependency") or {}).get("sensors") or {})
    imgs = {}
    for cam in CAM_ORDER:
        cam_doc = sensors.get(cam)
        if not cam_doc:
            continue
        img = load_camera_image(cam_doc)
        if img is not None:
            imgs[cam] = img
    return xyz, sensors, imgs


def draw_header(width: int, lines: list[str]) -> np.ndarray:
    panel = np.full((HEADER_H, width, 3), 28, np.uint8)
    y = 28
    for i, line in enumerate(lines):
        for chunk in textwrap.wrap(line, width=110) or [""]:
            color = (0, 220, 255) if i < 2 else (220, 220, 220)
            cv2.putText(panel, chunk, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(panel, chunk, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
            y += 26
            if y > HEADER_H - 10:
                return panel
    return panel


def compose_clip(
    clip_id: str,
    row: dict,
    bag_name: str,
    metrics: dict | None,
) -> dict:
    frames = ls.fetch_records([clip_id])[0]["frames"]
    if len(frames) < 5:
        return {"status": "no_frames"}
    wi, src = pick_anchor(frames, metrics)
    if frames[wi][1] is None:
        return {"status": "no_pose"}
    idxs = neighbor_idxs(len(frames), wi, N_AGG)
    idxs = [i for i in idxs if frames[i][1] is not None and frames[i][3]]
    pts, fids, traj_win = aggregate_map_clouds(clip_id, frames, idxs)
    # full traj for context
    traj_all = np.array([f[1] for f in frames if f[1] is not None], float)
    anchor = np.asarray(frames[wi][1][:2], float)
    box = suspect_box_xy(traj_win, anchor)

    bev_full = render_bev(
        pts, fids, traj_all, traj_win, anchor, box, zoom=False,
        title="BEV full (red box = suspect)", width=1100, height=1100,
    )
    bev_zoom = render_bev(
        pts, fids, traj_all, traj_win, anchor, box, zoom=True,
        title="BEV ZOOM suspect region", width=1100, height=1100,
    )
    bev_row = np.hstack([bev_full, bev_zoom])

    # cameras
    xyz_veh, sensors, imgs = get_anchor_xyz_and_cams(clip_id, frames, wi)
    cam_panels = []
    for cam in CAM_ORDER:
        cam_panels.append(
            project_cam_panel(xyz_veh, cam, sensors.get(cam), imgs.get(cam))
        )
    # 2x3 grid
    row1 = np.hstack(cam_panels[:3])
    row2 = np.hstack(cam_panels[3:6])
    # match width to bev_row
    cams = np.vstack([row1, row2])
    target_w = bev_row.shape[1]
    scale = target_w / cams.shape[1]
    cams = cv2.resize(
        cams, (target_w, int(cams.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )

    flags = row.get("flags") or []
    en = explain_en(flags, {**row, **(metrics or {})})
    zh = explain_zh(flags, {**row, **(metrics or {})})
    header_lines = [
        f"bag_name: {bag_name}",
        f"clip_id:  {clip_id}",
        (
            f"tier={row.get('tier')}  cat={row.get('reason_cat')}  "
            f"score={row.get('suspicion_score')}  flags: {','.join(flags)}"
        ),
        *en,
        (
            f"anchor=f{wi}/{len(frames)-1} ({src})  agg={len(idxs)}frames  "
            f"box=[{box[0]:.1f},{box[1]:.1f}]-[{box[2]:.1f},{box[3]:.1f}]m  "
            f"pts={len(pts):,}"
        ),
    ]
    header = draw_header(target_w, header_lines)
    canvas = np.vstack([header, bev_row, cams])
    return {
        "status": "ok",
        "image": canvas,
        "worst_i": wi,
        "anchor_src": src,
        "box": box,
        "n_agg": len(idxs),
        "n_points": int(len(pts)),
        "n_cams": sum(1 for c in CAM_ORDER if imgs.get(c) is not None),
        "explain_zh": zh,
    }


def write_index(summary: list[dict], out_dir: Path, title: str) -> None:
    from collections import Counter

    lines = [
        f"# {title}",
        "",
        "每张图自上而下：说明文字 → **BEV全景 + BEV可疑区放大（红框 SUSPECT）** → "
        "锚点帧 6 路相机 lidar 投影。",
        "",
        "- 按 **类别** 分组，组内按 **suspicion_score** 从高到低",
        "- BEV 颜色 = 帧序（彩虹分层/双线 = 鬼影）",
        "- 黄线 = 聚合窗口轨迹；灰线 = 全 clip 轨迹；黄十字 = 锚点",
        "- 红框 = 算法标出的可疑区域（请重点看框内）",
        "",
        f"总计: **{len(summary)}** 条",
        "",
        "## 类别汇总",
        "",
    ]
    for cat, n in Counter(s.get("reason_cat") or "NONE" for s in summary).most_common():
        lines.append(f"- `{cat}`: {n}")
    lines.append("")

    cur_cat = None
    for i, s in enumerate(summary, 1):
        cat = s.get("reason_cat") or "NONE"
        if cat != cur_cat:
            cur_cat = cat
            lines.append(f"## 类别 `{cat}`")
            lines.append("")
        lines.append(
            f"### {i}. score={s.get('suspicion_score')} `{s.get('bag_name', '')}`"
        )
        lines.append("")
        lines.append(f"- **bag_name**: `{s.get('bag_name', '')}`")
        lines.append(f"- **clip_id**: `{s.get('clip_id', '')}`")
        lines.append(
            f"- **tier / cat / score**: `{s.get('tier')}` / `{cat}` / `{s.get('suspicion_score')}`"
        )
        lines.append(f"- **flags**: `{','.join(s.get('flags') or [])}`")
        if s.get("status") != "ok":
            lines.append(f"- **status**: `{s.get('status')}`")
            lines.append("")
            continue
        rel = Path(s["out"]).name
        lines.append(f"- **image**: [{rel}](./{rel})")
        lines.append(
            f"- **anchor**: f{s.get('worst_i')} ({s.get('anchor_src')}), "
            f"cams={s.get('n_cams')}, pts={s.get('n_points')}"
        )
        lines.append("")
        lines.append("**说明**")
        for e in s.get("explain_zh") or []:
            lines.append(f"- {e}")
        lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    from collections import defaultdict

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--out-dir", type=Path, default=OUTDIR)
    ap.add_argument(
        "--tiers",
        nargs="+",
        default=["REJECT"],
        help="tiers to render, e.g. REJECT or HIGH WARN",
    )
    ap.add_argument(
        "--list",
        type=Path,
        default=LIST,
        help="final_badcase_list.json",
    )
    ap.add_argument(
        "--clean-glob",
        default="*.png",
        help="delete matching files in out-dir before render",
    )
    args = ap.parse_args()

    rows = json.loads(args.list.read_text())
    want = {t.upper() for t in args.tiers}
    rows = [r for r in rows if (r.get("tier") or "").upper() in want]
    if args.clips:
        want_ids = set(args.clips)
        rows = [
            r
            for r in rows
            if r["clip_id"] in want_ids or r["clip_id"][:8] in want_ids
        ]

    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r.get("reason_cat") or "NONE"].append(r)
    cat_order = sorted(
        by_cat.keys(),
        key=lambda c: -(
            max((x.get("suspicion_score") or 0) for x in by_cat[c]) if by_cat[c] else 0
        ),
    )
    ordered: list[dict] = []
    for c in cat_order:
        by_cat[c].sort(key=lambda x: -(x.get("suspicion_score") or 0))
        ordered.extend(by_cat[c])
    rows = ordered
    if args.limit:
        rows = rows[: args.limit]

    metrics_map = load_metrics()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob(args.clean_glob):
        if old.is_file():
            old.unlink()

    bags = fetch_bag_meta([r["clip_id"] for r in rows])
    title = f"Suspect triage ({'+'.join(sorted(want))}) — score/category"
    print(f"{len(rows)} clips {sorted(want)} → {args.out_dir}", flush=True)
    summary = []
    for i, row in enumerate(rows):
        cid = row["clip_id"]
        bag = (bags.get(cid) or {}).get("bag_name") or "UNKNOWN_BAG"
        cat = row.get("reason_cat") or "NONE"
        score = row.get("suspicion_score")
        try:
            info = compose_clip(cid, row, bag, metrics_map.get(cid))
        except Exception as exc:
            info = {"status": f"ERR:{type(exc).__name__}:{exc}"[:180]}
        if info.get("status") == "ok":
            sc = f"{float(score):.1f}" if score is not None else "na"
            out = (
                args.out_dir
                / f"{i+1:04d}_{row.get('tier', 'X')}_{cat}_s{sc}_{cid}_f{info['worst_i']}.png"
            )
            cv2.imwrite(str(out), info["image"])
            info["out"] = str(out)
            del info["image"]
            msg = (
                f"{bag} f{info['worst_i']} cams={info['n_cams']} "
                f"pts={info['n_points']:,} score={score}"
            )
        else:
            msg = str(info.get("status"))
        info["clip_id"] = cid
        info["bag_name"] = bag
        info["flags"] = row.get("flags")
        info["tier"] = row.get("tier")
        info["reason_cat"] = cat
        info["suspicion_score"] = score
        if "explain_zh" not in info:
            info["explain_zh"] = explain_zh(row.get("flags") or [], row)
        summary.append(info)
        print(f"[{i+1}/{len(rows)}] {cid}  {msg}", flush=True)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_index(summary, args.out_dir, title)
    n_ok = sum(1 for s in summary if s.get("status") == "ok")
    print(f"DONE {n_ok}/{len(rows)}  {args.out_dir / 'index.md'}", flush=True)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
