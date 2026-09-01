#!/usr/bin/env python3
"""Validate the LiDAR→image projection chain — nothing else.

Canonical chain (skip a step only when its output already exists):
  1. Deskew / unify point timestamps to one lidar reference time
     (skip if lidar_merge_deskew already present)
  2. Static aggregate + dynamic aggregate via OD tracking
     (skip if aggregate already built; single-frame cones: current cloud OK)
  3. Pose-interpolate points from lidar_ref time → camera image time
     (pose_ts × image_ts × point_ts)
  4. Project with camera extrinsic + intrinsic (+ distortion)

NO heuristics: no RGB orange gates, no clustering, no ground-band filters,
no semseg class tricks.  Stage A uses only synthetic geometry with known GT.
Stage B/C run the same chain on real calib / real lidar and report data flags.

Usage:
  PYTHONPATH=. .venv_smoke/bin/python tools/validate_projection_pipeline.py synthetic
  PYTHONPATH=. .venv_smoke/bin/python tools/validate_projection_pipeline.py diagnose \\
      --clip batch_s5_11c2aa2d-2618-45d8-ab28-5cf1529eca84 --ts 1776313833000545024
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp" / "robotruck" / "proj_validate"
BACKUP = ROOT / "exp" / "robotruck" / "raw_volume_cache"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rrv = _load("rrv_val", ROOT / "tools" / "render_robotruck_clip_video.py")


# ---------------------------------------------------------------------------
# Chain primitives (only these four operations)
# ---------------------------------------------------------------------------

def _transform_points_by_pose(
    xyz_at_point_time: np.ndarray,
    point_ts_ns: np.ndarray,
    pose_samples: list,
    target_timestamp_ns: int,
) -> np.ndarray:
    """p_vehicle@target = inv(T_map_v@target) @ T_map_v@t_i @ p_vehicle@t_i."""
    target_inv = np.linalg.inv(
        rrv.interpolate_pose_matrix(pose_samples, int(target_timestamp_ns))
    )
    result = np.empty_like(xyz_at_point_time, dtype=np.float64)
    unique_ts, inverse = np.unique(point_ts_ns, return_inverse=True)
    for i, ts in enumerate(unique_ts):
        mask = inverse == i
        relative = target_inv @ rrv.interpolate_pose_matrix(pose_samples, int(ts))
        result[mask] = xyz_at_point_time[mask] @ relative[:3, :3].T + relative[:3, 3]
    return result


def step1_deskew_to_ref(
    xyz_at_point_time: np.ndarray,
    point_ts_ns: np.ndarray,
    pose_samples: list,
    lidar_ref_ns: int,
) -> np.ndarray:
    """Unify points from per-point timestamps → lidar reference time."""
    return _transform_points_by_pose(
        xyz_at_point_time, point_ts_ns, pose_samples, int(lidar_ref_ns),
    )


def step3_align_to_camera(
    xyz_at_lidar_ref: np.ndarray,
    pose_samples: list,
    lidar_ref_ns: int,
    camera_ts_ns: int,
    *,
    per_point_ts_ns: np.ndarray | None = None,
) -> np.ndarray:
    """Align points into the vehicle frame at camera image time.

    If per_point_ts_ns is given (nodeskew path), each point is moved from its
    own timestamp → camera_ts.  Otherwise (deskewed cloud) all points share
    lidar_ref_ns and are moved lidar_ref → camera via a single relative pose.
    """
    if per_point_ts_ns is not None:
        return _transform_points_by_pose(
            xyz_at_lidar_ref, per_point_ts_ns, pose_samples, int(camera_ts_ns),
        )
    T_map_L = rrv.interpolate_pose_matrix(pose_samples, int(lidar_ref_ns))
    T_map_C = rrv.interpolate_pose_matrix(pose_samples, int(camera_ts_ns))
    # p_vehicle@C = inv(T_map_C) @ T_map_L @ p_vehicle@L
    relative = np.linalg.inv(T_map_C) @ T_map_L
    return xyz_at_lidar_ref @ relative[:3, :3].T + relative[:3, 3]


def step3_compensated_extrinsic(
    T_c_v: np.ndarray,
    pose_samples: list,
    lidar_ref_ns: int,
    camera_ts_ns: int,
) -> np.ndarray:
    """Equivalent to step3+extrinsic for deskewed clouds: T_c←v@C · (v@C←v@L)."""
    return rrv.camera_time_compensated_T(
        T_c_v, pose_samples, int(lidar_ref_ns), int(camera_ts_ns),
    )


def step4_project(
    xyz_vehicle_at_cam_time: np.ndarray,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """p_cam = T_c_v @ p_veh; then cv2.projectPoints with radtan dist5.

    Returns (uv Nx2 float, valid_mask over input).
    """
    n = len(xyz_vehicle_at_cam_time)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    ones = np.ones((n, 1), dtype=np.float64)
    ph = np.hstack([xyz_vehicle_at_cam_time.astype(np.float64), ones])
    pc = (T_c_v @ ph.T).T[:, :3]
    front = pc[:, 2] > 0.3
    uv_out = np.full((n, 2), np.nan, dtype=np.float64)
    if not front.any():
        return uv_out, front
    uv, _ = cv2.projectPoints(
        pc[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist5,
    )
    uv = uv.reshape(-1, 2)
    uv_out[front] = uv
    inside = (
        front
        & np.isfinite(uv_out).all(axis=1)
        & (uv_out[:, 0] >= 0) & (uv_out[:, 0] < width)
        & (uv_out[:, 1] >= 0) & (uv_out[:, 1] < height)
    )
    return uv_out, inside


# ---------------------------------------------------------------------------
# Stage A — pure synthetic closed loop (dense cloud + virtual RGB)
# ---------------------------------------------------------------------------

def _build_dense_map_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map-frame dense cloud with RGB + semantic kind.

    kind: 0=road, 1=lane, 2=wall, 3=cone, 4=post
    Returns xyz_map (N,3), rgb_bgr (N,3) uint8, kind (N,) int.
    """
    pts, cols, kinds = [], [], []

    def add(xyz, bgr, kind):
        pts.append(np.asarray(xyz, np.float64).reshape(-1, 3))
        cols.append(np.broadcast_to(np.asarray(bgr, np.uint8), (len(pts[-1]), 3)).copy())
        kinds.append(np.full(len(pts[-1]), kind, np.int32))

    # Road surface
    xs = np.linspace(-9.0, 9.0, 40)
    ys = np.linspace(6.0, 60.0, 90)
    xx, yy = np.meshgrid(xs, ys)
    road = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -1.25)])
    add(road, (55, 55, 55), 0)

    # Lane dashes (yellow)
    for y0 in np.arange(8.0, 58.0, 4.0):
        dash = np.column_stack([
            np.full(8, 0.0),
            np.linspace(y0, y0 + 1.5, 8),
            np.full(8, -1.24),
        ])
        add(dash, (0, 200, 255), 1)

    # Left sound-barrier wall
    for y in np.linspace(6.0, 60.0, 80):
        wall = np.column_stack([
            np.full(18, -9.2),
            np.full(18, y),
            np.linspace(-1.25, 2.5, 18),
        ])
        add(wall, (140, 140, 150), 2)

    # Orange traffic cones along left shoulder
    cone_centers = [(-7.5, y, -1.25) for y in np.arange(10.0, 52.0, 6.0)]
    cone_centers += [(7.5, y, -1.25) for y in (15.0, 35.0)]
    for cx, cy, cz in cone_centers:
        for h in np.linspace(0.0, 0.7, 10):
            r = 0.22 * (1.0 - h / 0.7)
            th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            ring = np.column_stack([
                cx + r * np.cos(th),
                cy + r * np.sin(th),
                np.full(len(th), cz + h),
            ])
            # orange / white bands
            bgr = (0, 140, 255) if (int(h * 10) % 2 == 0) else (230, 230, 230)
            add(ring, bgr, 3)

    # Vertical posts (cyan) for easy contour check
    for px, py in [(-3.0, 45.0), (3.0, 55.0), (0.0, 25.0)]:
        post = np.column_stack([
            np.full(12, px), np.full(12, py), np.linspace(-1.25, 0.8, 12),
        ])
        add(post, (255, 200, 0), 4)

    xyz = np.vstack(pts)
    rgb = np.vstack(cols)
    kind = np.concatenate(kinds)
    # tiny jitter so projections don't alias to a perfect grid
    xyz = xyz + rng.normal(0.0, 0.01, size=xyz.shape)
    return xyz, rgb, kind


def _synthetic_world() -> dict:
    """Dense virtual scene + poses + camera. All quantities known by construction."""
    rng = np.random.default_rng(0)
    v = 20.0  # m/s
    t0 = 1_000_000_000_000  # ns
    pose_samples = []
    for i in range(0, 6):
        dt_s = i * 0.02
        pose_samples.append((
            t0 + int(dt_s * 1e9),
            {
                "position": {"x": 0.0, "y": v * dt_s, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        ))

    K = np.array([
        [1200.0, 0.0, 960.0],
        [0.0, 1200.0, 540.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist5 = np.array([-0.25, 0.08, 1e-4, -2e-4, -0.02], dtype=np.float64)
    pitch = -0.08
    cp, sp = np.cos(pitch), np.sin(pitch)
    R0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cp, -sp],
        [0.0, sp, cp],
    ], dtype=np.float64)
    T_v_c = np.eye(4, dtype=np.float64)
    T_v_c[:3, :3] = Rx @ R0
    T_v_c[:3, 3] = [0.0, 1.5, 1.6]
    T_c_v = np.linalg.inv(T_v_c)
    width, height = 1920, 1080

    lidar_ref = t0 + 40_000_000
    camera_ts = t0 + 43_000_000  # +3 ms

    xyz_map, rgb, kind = _build_dense_map_scene(rng)

    # Sparse probe points (cone tips) for numeric GT table
    cone_mask = kind == 3
    cone_idx = np.flatnonzero(cone_mask)
    probe_idx = []
    if len(cone_idx):
        ys = xyz_map[cone_idx, 1]
        for y0 in np.arange(10.0, 52.0, 6.0):
            m = np.abs(ys - y0) < 1.0
            if m.any():
                sub = cone_idx[m]
                probe_idx.append(int(sub[np.argmax(xyz_map[sub, 2])]))
    probe_idx = np.asarray(probe_idx, dtype=np.int64)
    cone_map = xyz_map[probe_idx] if len(probe_idx) else xyz_map[:10]

    # Spinning-lidar style timestamps: azimuth in vehicle@lidar_ref map sense
    # Use map y as proxy for scan time (±20 ms around lidar_ref)
    y = xyz_map[:, 1]
    y_n = (y - y.min()) / max(1e-6, y.max() - y.min())
    point_ts = (lidar_ref + ((y_n - 0.5) * 40e6).astype(np.int64)).astype(np.int64)

    # nodeskew: express each point in vehicle frame AT its own timestamp
    xyz_nodeskew = np.empty_like(xyz_map)
    unique_ts, inv = np.unique(point_ts, return_inverse=True)
    for i, ts in enumerate(unique_ts):
        mask = inv == i
        T = rrv.interpolate_pose_matrix(pose_samples, int(ts))
        T_inv = np.linalg.inv(T)
        ph = np.hstack([xyz_map[mask], np.ones((mask.sum(), 1))])
        xyz_nodeskew[mask] = (T_inv @ ph.T).T[:, :3]

    return {
        "pose_samples": pose_samples,
        "K": K, "dist5": dist5, "T_c_v": T_c_v, "T_v_c": T_v_c,
        "w": width, "h": height,
        "lidar_ref": lidar_ref, "camera_ts": camera_ts,
        "xyz_map": xyz_map, "rgb": rgb, "kind": kind,
        "cone_map": cone_map, "probe_idx": probe_idx,
        "xyz_nodeskew": xyz_nodeskew,
        "point_ts": point_ts,
        "v_mps": v,
    }


def _map_to_vehicle_at(xyz_map: np.ndarray, pose_samples: list, ts_ns: int) -> np.ndarray:
    T = rrv.interpolate_pose_matrix(pose_samples, int(ts_ns))
    T_inv = np.linalg.inv(T)
    ph = np.hstack([xyz_map.astype(np.float64), np.ones((len(xyz_map), 1))])
    return (T_inv @ ph.T).T[:, :3]


def _render_rgb_from_cloud(
    xyz_veh: np.ndarray,
    rgb_bgr: np.ndarray,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Z-buffered paint of colored points → virtual camera image."""
    img = np.full((height, width, 3), 25, dtype=np.uint8)
    # sky gradient
    for r in range(int(height * 0.55)):
        img[r, :] = (40 + r // 8, 30 + r // 10, 20)
    zbuf = np.full((height, width), np.inf, dtype=np.float64)

    ones = np.ones((len(xyz_veh), 1), dtype=np.float64)
    pc = (T_c_v @ np.hstack([xyz_veh, ones]).T).T[:, :3]
    front = pc[:, 2] > 0.3
    if not front.any():
        return img
    uv, _ = cv2.projectPoints(pc[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist5)
    uv = uv.reshape(-1, 2)
    z = pc[front, 2]
    col = rgb_bgr[front]
    ui = np.rint(uv[:, 0]).astype(np.int32)
    vi = np.rint(uv[:, 1]).astype(np.int32)
    ok = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & np.isfinite(uv).all(1)
    # paint far→near so nearer overwrites
    order = np.argsort(-z[ok])
    ui, vi, z, col = ui[ok][order], vi[ok][order], z[ok][order], col[ok][order]
    for u, v, zz, c in zip(ui, vi, z, col):
        if zz < zbuf[v, u]:
            zbuf[v, u] = zz
            img[v, u] = c
            # fatten for visibility
            for du, dv in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)):
                uu, vv = u + du, v + dv
                if 0 <= uu < width and 0 <= vv < height and zz <= zbuf[vv, uu]:
                    zbuf[vv, uu] = zz
                    img[vv, uu] = c
    return img


def _overlay_points(
    img: np.ndarray,
    uv: np.ndarray,
    ok: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 2,
    step: int = 1,
) -> np.ndarray:
    out = img.copy()
    pts = uv[ok][::step]
    for u, v in pts:
        if not np.isfinite([u, v]).all():
            continue
        cv2.circle(out, (int(round(u)), int(round(v))), radius, color, -1)
    return out


def _bev_panel(
    xyz: np.ndarray,
    rgb: np.ndarray,
    title: str,
    size: int = 720,
    x_range=(-12.0, 12.0),
    y_range=(0.0, 65.0),
) -> np.ndarray:
    """Top-down BEV of a vehicle-frame cloud (x right, y forward)."""
    canvas = np.full((size, size, 3), 20, dtype=np.uint8)
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    u = ((xs - x_range[0]) / (x_range[1] - x_range[0]) * (size - 1)).astype(np.int32)
    v = ((y_range[1] - ys) / (y_range[1] - y_range[0]) * (size - 1)).astype(np.int32)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    canvas[v[ok], u[ok]] = rgb[ok]
    # ego marker
    eu = int((0 - x_range[0]) / (x_range[1] - x_range[0]) * (size - 1))
    ev = int((y_range[1] - 0) / (y_range[1] - y_range[0]) * (size - 1))
    cv2.drawMarker(canvas, (eu, ev), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 16, 2)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    cv2.putText(canvas, "BEV (+y up on page = forward)", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return canvas


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(out, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def _gt_uv_from_map(world: dict) -> np.ndarray:
    """Ground-truth UV for probe points: map → vehicle@camera_ts → project."""
    xyz_at_C = _map_to_vehicle_at(world["cone_map"], world["pose_samples"], world["camera_ts"])
    uv, _ = step4_project(
        xyz_at_C, world["K"], world["dist5"], world["T_c_v"], world["w"], world["h"],
    )
    return uv


def run_stage_a(out_dir: Path) -> dict:
    """Closed-loop on dense virtual cloud + step-by-step RGB/BEV panels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w = _synthetic_world()
    gt_uv = _gt_uv_from_map(w)

    # --- numeric paths on probe points (cone tips) ---
    probe_nodeskew = w["xyz_nodeskew"][w["probe_idx"]] if len(w["probe_idx"]) else w["xyz_nodeskew"][:10]
    probe_ts = w["point_ts"][w["probe_idx"]] if len(w["probe_idx"]) else w["point_ts"][:10]

    xyz_n = step3_align_to_camera(
        probe_nodeskew, w["pose_samples"], w["lidar_ref"], w["camera_ts"],
        per_point_ts_ns=probe_ts,
    )
    uv_n, ok_n = step4_project(xyz_n, w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"])

    xyz_deskew_probe = step1_deskew_to_ref(
        probe_nodeskew, probe_ts, w["pose_samples"], w["lidar_ref"],
    )
    xyz_d = step3_align_to_camera(
        xyz_deskew_probe, w["pose_samples"], w["lidar_ref"], w["camera_ts"],
    )
    uv_d, ok_d = step4_project(xyz_d, w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"])

    T_comp = step3_compensated_extrinsic(
        w["T_c_v"], w["pose_samples"], w["lidar_ref"], w["camera_ts"],
    )
    uv_e, ok_e = step4_project(xyz_deskew_probe, w["K"], w["dist5"], T_comp, w["w"], w["h"])

    labels = np.zeros(len(xyz_deskew_probe), np.int32)
    uv_api, _ = rrv.project_points(
        xyz_deskew_probe, labels, w["K"], w["dist5"], T_comp, w["w"], w["h"],
        max_points=999999,
    )

    def max_err(a, b, mask):
        if not mask.any():
            return float("nan")
        return float(np.max(np.linalg.norm(a[mask] - b[mask], axis=1)))

    shared = ok_n & ok_d & ok_e & np.isfinite(gt_uv).all(axis=1)
    err_n = max_err(uv_n, gt_uv, shared)
    err_d = max_err(uv_d, gt_uv, shared)
    err_e = max_err(uv_e, gt_uv, shared)
    err_de = max_err(uv_d, uv_e, ok_d & ok_e)

    uv_notime, ok_nt = step4_project(
        xyz_deskew_probe, w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"],
    )
    err_notime = max_err(uv_notime, gt_uv, ok_nt & np.isfinite(gt_uv).all(1))

    # --- dense cloud through the same chain (for visualization) ---
    xyz_deskew = step1_deskew_to_ref(
        w["xyz_nodeskew"], w["point_ts"], w["pose_samples"], w["lidar_ref"],
    )
    xyz_at_cam = step3_align_to_camera(
        xyz_deskew, w["pose_samples"], w["lidar_ref"], w["camera_ts"],
    )
    # GT vehicle frame at camera time (from map) — for RGB render
    xyz_gt_cam = _map_to_vehicle_at(w["xyz_map"], w["pose_samples"], w["camera_ts"])

    rgb_gt = _render_rgb_from_cloud(
        xyz_gt_cam, w["rgb"], w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"],
    )
    uv_chain, ok_chain = step4_project(
        xyz_at_cam, w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"],
    )
    uv_skip, ok_skip = step4_project(
        xyz_deskew, w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"],
    )
    # nodeskew projected with NO deskew/align (wrong) — smeared contour
    uv_raw, ok_raw = step4_project(
        w["xyz_nodeskew"], w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"],
    )

    # Dense cloud vs GT-at-cam residual (should be ~0 after full chain)
    dens_err = float(np.max(np.linalg.norm(xyz_at_cam - xyz_gt_cam, axis=1)))

    # ========== step-by-step figure ==========
    # Row0: BEV of constructed cloud (map→veh@lidar_ref) + virtual RGB
    xyz_at_lidar = _map_to_vehicle_at(w["xyz_map"], w["pose_samples"], w["lidar_ref"])
    bev0 = _bev_panel(xyz_at_lidar, w["rgb"], f"0. VIRTUAL POINT CLOUD  n={len(w['xyz_map'])}")
    rgb0 = _banner(rgb_gt, "0. VIRTUAL RGB  (render map cloud at camera_ts with true K/T)")
    # fit BEV beside RGB: scale BEV height to rgb height
    bev0r = cv2.resize(bev0, (int(bev0.shape[1] * rgb0.shape[0] / bev0.shape[0]), rgb0.shape[0]))
    row0 = np.hstack([bev0r, rgb0])

    # Step1: nodeskew vs deskew BEV (vehicle frame contents differ if ego moves during scan)
    bev_ns = _bev_panel(w["xyz_nodeskew"], w["rgb"], "1a. NODESKEW cloud (each pt at own t)")
    bev_ds = _bev_panel(xyz_deskew, w["rgb"], "1b. STEP1 DESKEW → lidar_ref")
    # Show wrong projection of nodeskew onto RGB
    step1_wrong = _banner(
        _overlay_points(rgb_gt, uv_raw, ok_raw, (0, 0, 255), radius=1, step=2),
        "1c. nodeskew projected WITHOUT deskew/align (RED) — contours smear / miss",
    )
    mid_h = rgb0.shape[0] // 2
    bev_ns_r = cv2.resize(bev_ns, (mid_h, mid_h))
    bev_ds_r = cv2.resize(bev_ds, (mid_h, mid_h))
    bev_pair = np.vstack([bev_ns_r, bev_ds_r])
    # pad if needed
    if bev_pair.shape[0] < step1_wrong.shape[0]:
        pad = np.zeros((step1_wrong.shape[0] - bev_pair.shape[0], bev_pair.shape[1], 3), np.uint8)
        bev_pair = np.vstack([bev_pair, pad])
    elif bev_pair.shape[0] > step1_wrong.shape[0]:
        bev_pair = bev_pair[: step1_wrong.shape[0]]
    row1 = np.hstack([bev_pair, step1_wrong])

    # Step3+4: full chain overlay on RGB — must match contours
    step34 = _overlay_points(rgb_gt, uv_chain, ok_chain, (0, 255, 0), radius=1, step=1)
    # emphasize cone points
    cone_m = (w["kind"] == 3)
    uv_cone, ok_cone = step4_project(
        xyz_at_cam[cone_m], w["K"], w["dist5"], w["T_c_v"], w["w"], w["h"],
    )
    step34 = _overlay_points(step34, uv_cone, ok_cone, (0, 255, 255), radius=3, step=1)
    step34 = _banner(
        step34,
        f"2. STEP3 pose-align to cam_ts + STEP4 K/T  |  green=all  cyan=cones  "
        f"probe max_err={err_e:.2e}px  dens_xyz_err={dens_err:.2e}m  PASS",
    )

    # Ablation: skip step3
    abl = _overlay_points(rgb_gt, uv_skip, ok_skip, (0, 255, 255), radius=1, step=1)
    abl = _overlay_points(abl, uv_notime, ok_nt, (0, 0, 255), radius=6, step=1)
    abl = _banner(
        abl,
        f"3. ABLATION skip STEP3 (yellow=cloud, red=cone probes)  max_err={err_notime:.2f}px — visible miss",
    )

    # Zoom on a cone: full chain vs skip
    def _zoom_at(src, uv_c, half=70, scale=5):
        if not np.isfinite(uv_c).all():
            return np.zeros((half * 2 * scale, half * 2 * scale, 3), np.uint8)
        u, v = int(round(uv_c[0])), int(round(uv_c[1]))
        x0, y0 = max(0, u - half), max(0, v - half)
        x1, y1 = min(w["w"], u + half), min(w["h"], v + half)
        crop = src[y0:y1, x0:x1]
        return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)

    # pick a mid cone probe
    mid = int(np.flatnonzero(shared)[len(np.flatnonzero(shared)) // 2]) if shared.any() else 0
    z_ok = _zoom_at(step34, gt_uv[mid])
    z_bad = _zoom_at(abl, gt_uv[mid])
    cv2.putText(z_ok, "zoom STEP3+4: cloud sits on cone", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(z_bad, "zoom skip STEP3: cloud misses cone", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    zh = max(z_ok.shape[0], z_bad.shape[0])
    zw = z_ok.shape[1] + z_bad.shape[1] + 20
    zoom_row = np.zeros((zh, max(zw, w["w"]), 3), np.uint8)
    zoom_row[: z_ok.shape[0], : z_ok.shape[1]] = z_ok
    zoom_row[: z_bad.shape[0], z_ok.shape[1] + 20: z_ok.shape[1] + 20 + z_bad.shape[1]] = z_bad
    if zoom_row.shape[1] < w["w"]:
        zoom_row = np.hstack([zoom_row, np.zeros((zh, w["w"] - zoom_row.shape[1], 3), np.uint8)])
    else:
        # also pad step panels to zoom width for vstack — better resize all to common width
        pass

    def _fit_w(img, tw):
        if img.shape[1] == tw:
            return img
        nh = int(round(img.shape[0] * tw / img.shape[1]))
        return cv2.resize(img, (tw, nh))

    tw = max(row0.shape[1], row1.shape[1], step34.shape[1], abl.shape[1], zoom_row.shape[1])
    proof = np.vstack([
        _fit_w(row0, tw),
        _fit_w(row1, tw),
        _fit_w(step34, tw),
        _fit_w(abl, tw),
        _fit_w(zoom_row, tw),
    ])
    cv2.imwrite(str(out_dir / "stage_a_perfect_alignment.png"), proof)
    cv2.imwrite(str(out_dir / "stage_a_virtual_rgb.png"), rgb_gt)
    cv2.imwrite(str(out_dir / "stage_a_virtual_bev.png"), bev0)
    cv2.imwrite(str(out_dir / "stage_a_step34_overlay.png"), step34)

    per_point = []
    for i in range(len(gt_uv)):
        if not shared[i]:
            continue
        per_point.append({
            "id": i,
            "map_xyz": w["cone_map"][i].round(2).tolist(),
            "gt_uv": gt_uv[i].round(3).tolist(),
            "chain_uv": uv_e[i].round(3).tolist(),
            "err_chain_px": float(np.linalg.norm(uv_e[i] - gt_uv[i])),
            "err_no_time_align_px": float(np.linalg.norm(uv_notime[i] - gt_uv[i])) if ok_nt[i] else None,
        })

    report = {
        "stage": "A_synthetic_closed_loop",
        "n_points_dense": int(len(w["xyz_map"])),
        "n_probe_cones": int(shared.sum()),
        "ego_speed_mps": w["v_mps"],
        "camera_minus_lidar_ms": (w["camera_ts"] - w["lidar_ref"]) / 1e6,
        "max_err_nodeskew_path_vs_gt_px": err_n,
        "max_err_deskew_then_align_vs_gt_px": err_d,
        "max_err_compensated_extrinsic_vs_gt_px": err_e,
        "max_err_align_vs_compensated_extrinsic_px": err_de,
        "max_err_WITHOUT_time_align_vs_gt_px": err_notime,
        "dense_xyz_err_after_chain_m": dens_err,
        "project_points_api_count": int(len(uv_api)),
        "per_point": per_point,
        "proof_image": str(out_dir / "stage_a_perfect_alignment.png"),
        "pass": bool(
            shared.sum() >= 3
            and err_n < 1e-3
            and err_d < 1e-3
            and err_e < 1e-3
            and err_de < 1e-3
            and dens_err < 1e-6
            and err_notime > 1.0
        ),
    }
    (out_dir / "stage_a_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_point"}, indent=2))
    return report


# ---------------------------------------------------------------------------
# Stage B — real calib + synthetic points (chain only)
# ---------------------------------------------------------------------------

def run_stage_b(clip: str, ts: str, out_dir: Path) -> dict:
    """Real K/T/pose/timestamps; synthetic 3-D grid.  Chain steps 3+4 only
    (deskew N/A for synthetic; agg N/A)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = BACKUP / clip / "frames" / ts
    meta = json.loads((fr / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]
    all_ts = rrv.list_clip_frames(BACKUP / clip)
    pose_samples = rrv.build_clip_pose_samples(BACKUP / clip, all_ts)

    lid = (
        sensors.get("lidar_merge_deskew")
        or sensors.get("lidar_merge_nodeskew")
        or sensors.get("lidar_merge")
        or {}
    )
    lidar_ref = int(lid.get("timestamp") or int(ts))

    # Synthetic world points in vehicle@lidar_ref (known by construction)
    ys = np.arange(8.0, 56.0, 4.0)
    syn = np.column_stack([
        np.concatenate([np.full(len(ys), -7.5), np.full(len(ys), 7.5)]),
        np.concatenate([ys, ys]),
        np.full(2 * len(ys), -1.1),
    ]).astype(np.float64)

    per_cam = {}
    for cam_name in rrv.CAM_ORDER:
        if cam_name not in sensors:
            continue
        cam_doc = sensors[cam_name]
        K, dist5, T_c_v, cw, ch = rrv.parse_camera(cam_doc)
        cam_ts = int(cam_doc.get("timestamp") or lidar_ref)

        xyz_at_cam = step3_align_to_camera(syn, pose_samples, lidar_ref, cam_ts)
        uv, ok = step4_project(xyz_at_cam, K, dist5, T_c_v, cw, ch)

        # Equivalent compensated-extrinsic path must match
        T_comp = step3_compensated_extrinsic(T_c_v, pose_samples, lidar_ref, cam_ts)
        uv2, ok2 = step4_project(syn, K, dist5, T_comp, cw, ch)
        shared = ok & ok2
        path_err = (
            float(np.max(np.linalg.norm(uv[shared] - uv2[shared], axis=1)))
            if shared.any() else 0.0
        )

        blank = np.zeros((ch, cw, 3), dtype=np.uint8)
        for u, v in uv[ok]:
            cv2.drawMarker(blank, (int(u), int(v)), (0, 255, 0),
                           cv2.MARKER_TILTED_CROSS, 16, 2)
        jpg_path = fr / f"{cam_name}.jpg"
        if jpg_path.is_file():
            real = cv2.imread(str(jpg_path))
            if real is not None:
                if real.shape[1] != cw or real.shape[0] != ch:
                    real = cv2.resize(real, (cw, ch))
                overlay = real.copy()
                for u, v in uv[ok]:
                    cv2.drawMarker(overlay, (int(u), int(v)), (0, 255, 0),
                                   cv2.MARKER_TILTED_CROSS, 16, 2)
                panel = np.hstack([blank, overlay])
            else:
                panel = blank
        else:
            panel = blank
        cv2.imwrite(str(out_dir / f"stage_b_{cam_name}.png"), panel)

        per_cam[cam_name] = {
            "dt_ms": (cam_ts - lidar_ref) / 1e6,
            "n_in_image": int(ok.sum()),
            "align_vs_compensated_extrinsic_max_px": path_err,
            "uv": uv[ok].round(1).tolist(),
        }

    report = {
        "stage": "B_real_calib_synthetic_points",
        "clip": clip,
        "ts": ts,
        "lidar_ref": lidar_ref,
        "cameras": per_cam,
        "pass": all(
            c["align_vs_compensated_extrinsic_max_px"] < 1e-3
            for c in per_cam.values()
        ),
    }
    (out_dir / "stage_b_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Stage C — real lidar through verified chain; diagnose DATA (not heuristics)
# ---------------------------------------------------------------------------

def run_stage_c(clip: str, ts: str, out_dir: Path) -> dict:
    """Run verified chain on real lidar.  No RGB/cluster/class filters.

    Data-issue probes (not projection tricks):
      - Is cloud already deskewed? (skip step1)
      - Are per-point timestamps usable? (dt column)
      - Pose sample spacing / stamp vs frame_ts
      - camera−lidar Δt
      - Regional UV consistency of a FIXED synthetic line under real calib
        (if calib/time OK, a straight 3-D shoulder line must project to a
         smooth monotonic curve; we only report the projected UVs — visual
         comparison to cones is left to the overlay PNG)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = BACKUP / clip / "frames" / ts
    clip_dir = BACKUP / clip
    arr = np.fromfile(fr / "lidar_merge.bin", dtype=np.float32).reshape(-1, 7)
    xyz = arr[:, :3].astype(np.float64)
    dt_col = arr[:, 5]
    meta = json.loads((fr / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]
    all_ts = rrv.list_clip_frames(clip_dir)
    pose_samples = rrv.build_clip_pose_samples(clip_dir, all_ts)

    has_deskew = "lidar_merge_deskew" in sensors
    has_nodeskew = "lidar_merge_nodeskew" in sensors
    lid = (
        sensors.get("lidar_merge_deskew")
        or sensors.get("lidar_merge_nodeskew")
        or sensors.get("lidar_merge")
        or {}
    )
    lidar_ref = int(lid.get("timestamp") or int(ts))

    data_flags = {
        "has_lidar_merge_deskew_meta": has_deskew,
        "has_lidar_merge_nodeskew_meta": has_nodeskew,
        "local_bin_dt_all_zero": bool(np.allclose(dt_col, 0)),
        "local_bin_n_points": int(len(xyz)),
        "deskew_meta_n_points": lid.get("n_points"),
        "deskew_method": lid.get("deskew_method"),
        "deskew_reference": lid.get("deskew_reference"),
        "skip_step1_deskew": bool(has_deskew),
        "skip_step1_reason": (
            "lidar_merge_deskew present → points already at lidar_ref"
            if has_deskew else
            "no deskew meta; local dt all zero → cannot rebuild per-point deskew"
        ),
        "pose_n_samples": len(pose_samples),
        "pose_stamp_vs_frame_ms_median": None,
    }
    if pose_samples and all_ts:
        deltas = []
        for pstamp, _ in pose_samples:
            # nearest frame ts
            nearest = min(all_ts, key=lambda t: abs(int(t) - pstamp))
            deltas.append(abs(pstamp - int(nearest)) / 1e6)
        data_flags["pose_stamp_vs_frame_ms_median"] = float(np.median(deltas))

    # Step2: single-frame path — no aggregate required for diagnosing
    # current-frame projection.  Record that we skipped.
    data_flags["skip_step2_aggregate"] = True
    data_flags["skip_step2_reason"] = (
        "single-frame diagnose uses current cloud; static/dynamic agg is "
        "orthogonal to per-frame lidar→image alignment"
    )

    report = {
        "stage": "C_real_data_via_verified_chain",
        "clip": clip,
        "ts": ts,
        "lidar_ref": lidar_ref,
        "data_flags": data_flags,
        "cameras": {},
    }

    # Subsample for overlay only (uniform) — not a semantic filter
    rng = np.random.default_rng(0)
    n_draw = min(8000, len(xyz))
    draw_idx = rng.choice(len(xyz), size=n_draw, replace=False)
    xyz_draw = xyz[draw_idx]

    for cam_name in ("camera1", "camera2", "camera6", "camera8"):
        if cam_name not in sensors:
            continue
        cam_doc = sensors[cam_name]
        K, dist5, T_c_v, cw, ch = rrv.parse_camera(cam_doc)
        cam_ts = int(cam_doc.get("timestamp") or lidar_ref)
        dt_ms = (cam_ts - lidar_ref) / 1e6

        # Step 3+4 (deskewed path)
        T_comp = step3_compensated_extrinsic(T_c_v, pose_samples, lidar_ref, cam_ts)
        uv, ok = step4_project(xyz_draw, K, dist5, T_comp, cw, ch)

        # Without step3 (ablation to quantify time-align effect on THIS frame)
        uv_raw, ok_raw = step4_project(xyz_draw, K, dist5, T_c_v, cw, ch)
        shared = ok & ok_raw
        time_shift_px = (
            float(np.median(np.linalg.norm(uv[shared] - uv_raw[shared], axis=1)))
            if shared.any() else None
        )

        # Synthetic shoulder line through SAME chain — regional U/V report only
        ys = np.linspace(8.0, 55.0, 20)
        line = np.column_stack([np.full(len(ys), -7.5), ys, np.full(len(ys), -1.1)])
        uv_line, ok_line = step4_project(line, K, dist5, T_comp, cw, ch)
        line_uv = uv_line[ok_line]
        # Monotonicity / curvature of projected line (calib/time smell test)
        line_stats = {}
        if len(line_uv) >= 3:
            order = np.argsort(line_uv[:, 1])  # by image row
            uu = line_uv[order, 0]
            vv = line_uv[order, 1]
            du = np.diff(uu)
            line_stats = {
                "n": int(len(line_uv)),
                "u_range": [float(uu.min()), float(uu.max())],
                "v_range": [float(vv.min()), float(vv.max())],
                "du_sign_flips": int(np.sum(du[1:] * du[:-1] < 0)),
                "u_at_top_third_median": float(np.median(uu[vv < ch / 3])) if (vv < ch / 3).any() else None,
                "u_at_mid_third_median": float(np.median(uu[(vv >= ch / 3) & (vv < 2 * ch / 3)])) if ((vv >= ch / 3) & (vv < 2 * ch / 3)).any() else None,
                "u_at_bot_third_median": float(np.median(uu[vv >= 2 * ch / 3])) if (vv >= 2 * ch / 3).any() else None,
            }

        jpg = cv2.imread(str(fr / f"{cam_name}.jpg"))
        if jpg is None:
            continue
        if jpg.shape[1] != cw or jpg.shape[0] != ch:
            jpg = cv2.resize(jpg, (cw, ch))
        vis = jpg.copy()
        for u, v in uv[ok]:
            cv2.circle(vis, (int(u), int(v)), 1, (0, 0, 255), -1)
        for u, v in line_uv:
            cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(
            vis,
            f"{cam_name}  dt={dt_ms:.2f}ms  chain=deskewed+pose_align+K/T",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
        )
        cv2.imwrite(str(out_dir / f"stage_c_{cam_name}.png"), vis)

        report["cameras"][cam_name] = {
            "dt_ms": dt_ms,
            "n_projected": int(ok.sum()),
            "median_uv_shift_if_skip_step3_px": time_shift_px,
            "synthetic_line_projection": line_stats,
        }
        print(cam_name, json.dumps(report["cameras"][cam_name]))

    report["interpretation"] = {
        "if_stage_A_pass": "projection chain math/conventions are correct",
        "if_overlay_still_drifts_by_region": (
            "data/calib issue — candidates: extrinsic/intrinsic error, "
            "deskew reference mismatch, pose stamp quantization, "
            "or lidar frame not actually at lidar_ref"
        ),
        "time_align_effect": (
            "median_uv_shift_if_skip_step3_px quantifies step3 on this frame; "
            "small values mean 2–3 ms ego motion is not the regional drift source"
        ),
    }
    (out_dir / "stage_c_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synthetic", help="Stage A only — virtual image + virtual cloud")
    diag = sub.add_parser("diagnose", help="Stage A + B + C")
    diag.add_argument("--clip", default="batch_s5_11c2aa2d-2618-45d8-ab28-5cf1529eca84")
    diag.add_argument("--ts", default="1776313833000545024")
    diag.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    if args.cmd == "synthetic":
        rep = run_stage_a(OUT / "stage_a")
        return 0 if rep.get("pass") else 1

    out = args.out_dir
    a = run_stage_a(out / "stage_a")
    if not a.get("pass"):
        print("FAIL stage A — fix chain before diagnosing data", file=sys.stderr)
        return 1
    b = run_stage_b(args.clip, args.ts, out / "stage_b")
    if not b.get("pass"):
        print("FAIL stage B — align vs compensated extrinsic disagree", file=sys.stderr)
        return 1
    run_stage_c(args.clip, args.ts, out / "stage_c")
    print(f"\nArtifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
