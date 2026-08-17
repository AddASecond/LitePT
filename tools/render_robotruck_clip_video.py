"""Render a Robotruck clip video: multi-cam + BEVs + sides + static agg + occupancy.

Each output frame is a composite of:
  - camera views with LitePT labels (clip-static + frame-dynamic)
  - BEV/side semantic and lidar_id panels
  - clip-level static aggregation via ego_pose
  - voxel occupancy panels (semantic / height / binary BEV + side YZ)

Usage:
  export PYTHONPATH=./
  .venv_smoke/bin/python tools/render_robotruck_clip_video.py \\
    --clip stop_1784423032302844849_vehicle-V002-20260719_090818 \\
    --stride 2 --fps 10 --reuse-pred --aggregate-static --occupancy
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_helpers():
    path = ROOT / "tools" / "infer_robotruck_mongo_frame.py"
    spec = importlib.util.spec_from_file_location("infer_robotruck_mongo_frame", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_h = _load_helpers()
from visualize import WAYMO_COLORS, WAYMO_NAMES  # noqa: E402


def _load_static_agg():
    path = ROOT / "tools" / "robotruck_static_agg.py"
    name = "robotruck_static_agg"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sag = _load_static_agg()


def _load_occupancy():
    path = ROOT / "tools" / "robotruck_occupancy.py"
    name = "robotruck_occupancy"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


occmod = _load_occupancy()

CAM_ORDER = [
    "camera1",
    "camera2",
    "camera3",
    "camera4",
    "camera5",
    "camera6",
    "camera7",
    "camera8",
    "camera9",
    "camera17",
]

# BGR colors for lidar_id distinction (1 / 2 / 14)
LIDAR_ID_BGR = {
    1: (255, 180, 40),  # blue-ish
    2: (40, 220, 40),  # green
    14: (40, 40, 255),  # red
}
LIDAR_ID_DEFAULT_BGR = (160, 160, 160)


def parse_camera(cam_doc: dict):
    K = np.asarray(cam_doc["intrinsic"]["intrinsic"], dtype=np.float64)
    dist = np.asarray(cam_doc["intrinsic"]["distortion"], dtype=np.float64).reshape(-1)
    dist5 = np.zeros(5, dtype=np.float64)
    dist5[: min(5, dist.size)] = dist[:5]
    # extrinsic.transformation is T_vehicle_camera (camera pose in vehicle)
    T_v_c = np.asarray(cam_doc["extrinsic"]["transformation"], dtype=np.float64)
    T_c_v = np.linalg.inv(T_v_c)
    w = int(cam_doc["intrinsic"]["width"])
    h = int(cam_doc["intrinsic"]["height"])
    return K, dist5, T_c_v, w, h


def labels_to_bgr(labels: np.ndarray) -> np.ndarray:
    """Map class ids to BGR uint8 for OpenCV."""
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(WAYMO_COLORS))
    rgb = (WAYMO_COLORS[labels[valid]] * 255.0).astype(np.uint8)
    colors[valid] = rgb[:, ::-1]  # RGB→BGR
    return colors


def project_points(
    xyz_veh: np.ndarray,
    labels: np.ndarray,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    width: int,
    height: int,
    max_points: int = 80000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uv int Nx2, bgr Nx3) for points visible in the camera."""
    n = xyz_veh.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        xyz_veh = xyz_veh[idx]
        labels = labels[idx]

    ones = np.ones((xyz_veh.shape[0], 1), dtype=np.float64)
    ph = np.hstack([xyz_veh.astype(np.float64), ones])
    pc = (T_c_v @ ph.T).T[:, :3]
    front = pc[:, 2] > 0.3
    pc = pc[front]
    labels = labels[front]
    if pc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)

    uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist5)
    uv = uv.reshape(-1, 2)
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
        & np.isfinite(uv).all(axis=1)
    )
    uv = uv[inside]
    labels = labels[inside]
    z = pc[inside][:, 2]
    order = np.argsort(-z)
    uv = uv[order]
    labels = labels[order]
    return uv.astype(np.int32), labels_to_bgr(labels)


def draw_projection(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    colors_bgr: np.ndarray,
    radius: int = 3,
    alpha: float = 0.85,
) -> np.ndarray:
    if uv.shape[0] == 0:
        return image_bgr
    overlay = image_bgr.copy()
    if uv.shape[0] > 100000:
        step = max(1, uv.shape[0] // 100000)
        uv = uv[::step]
        colors_bgr = colors_bgr[::step]
    for (uu, vv), c in zip(uv, colors_bgr):
        cv2.circle(
            overlay,
            (int(uu), int(vv)),
            radius,
            (int(c[0]), int(c[1]), int(c[2])),
            -1,
        )
    return cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)


def lidar_ids_to_bgr(lidar_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(lidar_ids).astype(np.int32).reshape(-1)
    colors = np.zeros((ids.shape[0], 3), dtype=np.uint8)
    colors[:] = LIDAR_ID_DEFAULT_BGR
    for lid, bgr in LIDAR_ID_BGR.items():
        colors[ids == lid] = bgr
    return colors


def render_plane_bgr(
    h_vals: np.ndarray,
    v_vals: np.ndarray,
    colors_bgr: np.ndarray,
    *,
    h_range: tuple[float, float],
    v_range: tuple[float, float],
    target_w: int,
    title: str = "",
    h_marks: tuple[float, ...] = (),
    h_mark_color: tuple[int, int, int] = (0, 200, 255),
    v_up: bool = True,
    max_points: int = 200000,
    seed: int = 0,
    legend_items: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """Isotropic scatter plane: horizontal h_vals, vertical v_vals (v_up → high at top)."""
    n = h_vals.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        h_vals = h_vals[idx]
        v_vals = v_vals[idx]
        colors_bgr = colors_bgr[idx]

    h0, h1 = h_range
    v0, v1 = v_range
    h_span = max(1e-6, h1 - h0)
    v_span = max(1e-6, v1 - v0)
    ppm = float(target_w) / h_span
    out_w = int(target_w)
    out_h = max(1, int(round(v_span * ppm)))

    u = ((h_vals - h0) * ppm).astype(np.int32)
    if v_up:
        vv = ((v1 - v_vals) * ppm).astype(np.int32)
    else:
        vv = ((v_vals - v0) * ppm).astype(np.int32)

    img = np.full((out_h, out_w, 3), 20, dtype=np.uint8)
    m = (u >= 0) & (u < out_w) & (vv >= 0) & (vv < out_h)
    img[vv[m], u[m]] = colors_bgr[m]

    def mh(meters: float) -> int:
        return int(round((meters - h0) * ppm))

    def mv(meters: float) -> int:
        if v_up:
            return int(round((v1 - meters) * ppm))
        return int(round((meters - v0) * ppm))

    for d in h_marks:
        uu = mh(d)
        if 0 <= uu < out_w:
            cv2.line(img, (uu, 0), (uu, out_h - 1), h_mark_color, 1)
            cv2.putText(
                img,
                f"{d:g}m",
                (uu + 2, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                h_mark_color,
                2,
            )

    if title:
        cv2.putText(
            img,
            f"{title}  {ppm:.2f}px/m",
            (12, out_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
        )

    if legend_items:
        x0, y0 = 12, 28
        for name, bgr in legend_items:
            cv2.rectangle(img, (x0, y0 - 14), (x0 + 22, y0 + 4), bgr, -1)
            cv2.putText(
                img,
                name,
                (x0 + 28, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                2,
            )
            x0 += 140
    return img


def render_bev_bgr(
    xyz: np.ndarray,
    colors_bgr: np.ndarray,
    *,
    x_range: tuple[float, float] = (-30.0, 30.0),
    y_range: tuple[float, float] = (-200.0, 400.0),
    target_w: int = 6400,
    title: str = "BEV",
    legend_items: list[tuple[str, tuple[int, int, int]]] | None = None,
    max_points: int = 200000,
    seed: int = 0,
    draw_roi: bool = True,
) -> np.ndarray:
    """Landscape BEV: +y forward→, +x right↓ (v_up=False so +x down)."""
    img = render_plane_bgr(
        xyz[:, 1],
        xyz[:, 0],
        colors_bgr,
        h_range=y_range,
        v_range=x_range,
        target_w=target_w,
        title=title,
        h_marks=(-200, -100, -50, 0, 50, 100, 150, 200, 250, 300, 350, 400),
        v_up=False,
        max_points=max_points,
        seed=seed,
        legend_items=legend_items,
    )
    if draw_roi:
        f0, f1 = y_range
        l0, l1 = x_range
        ppm = float(target_w) / max(1e-6, f1 - f0)
        out_h, out_w = img.shape[:2]

        def mf_to_u(meters: float) -> int:
            return int(round((meters - f0) * ppm))

        def ml_to_v(meters: float) -> int:
            return int(round((meters - l0) * ppm))

        for lat_m in (-24.0, 24.0):
            vv = ml_to_v(lat_m)
            if 0 <= vv < out_h:
                cv2.line(img, (mf_to_u(0), vv), (mf_to_u(min(400, f1)), vv), (255, 200, 100), 2)
        cv2.circle(img, (mf_to_u(0), ml_to_v(0)), max(3, int(ppm * 0.8)), (0, 0, 255), -1)
    return img


def render_side_views(
    xyz: np.ndarray,
    colors_bgr: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float] = (-5.0, 20.0),
    target_w: int,
    max_points: int = 200000,
    seed: int = 0,
    legend_items: list[tuple[str, tuple[int, int, int]]] | None = None,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (side_yz, side_xz) with shared isotropic ppm from YZ full width."""
    y0, y1 = y_range
    ppm = float(target_w) / max(1e-6, y1 - y0)
    tag = f"{label} " if label else ""
    # YZ: forward × height (full width)
    side_yz = render_plane_bgr(
        xyz[:, 1],
        xyz[:, 2],
        colors_bgr,
        h_range=y_range,
        v_range=z_range,
        target_w=target_w,
        title=f"Side YZ {tag}(+y forward {int(y0)}..{int(y1)}m, +z up)",
        h_marks=(-200, -100, 0, 100, 200, 300, 400),
        v_up=True,
        max_points=max_points,
        seed=seed,
        legend_items=legend_items,
    )
    # XZ: lateral × height (same ppm → width follows x_span)
    x0, x1 = x_range
    xz_w = max(1, int(round((x1 - x0) * ppm)))
    side_xz = render_plane_bgr(
        xyz[:, 0],
        xyz[:, 2],
        colors_bgr,
        h_range=x_range,
        v_range=z_range,
        target_w=xz_w,
        title=f"Side XZ {tag}(+x right {int(x0)}..{int(x1)}m, +z up)",
        h_marks=(-24, 0, 24),
        v_up=True,
        max_points=max_points,
        seed=seed + 1,
        legend_items=legend_items,
    )
    # Place XZ on a full-width canvas (same height as YZ), centered.
    canvas = np.full((side_yz.shape[0], target_w, 3), 20, dtype=np.uint8)
    if side_xz.shape[0] != canvas.shape[0]:
        side_xz = cv2.resize(
            side_xz, (side_xz.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    x_off = (target_w - side_xz.shape[1]) // 2
    canvas[:, x_off : x_off + side_xz.shape[1]] = side_xz
    cv2.putText(
        canvas,
        f"Side XZ {tag}(centered, same px/m as YZ)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
    )
    return side_yz, canvas


def resize_max(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if nh == h and nw == w:
        return img
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def fit_into(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Letterbox img into box, preserving aspect ratio (no stretch)."""
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    t = resize_max(img, box_w, box_h)
    y0 = (box_h - t.shape[0]) // 2
    x0 = (box_w - t.shape[1]) // 2
    canvas[y0 : y0 + t.shape[0], x0 : x0 + t.shape[1]] = t
    return canvas


def format_lidar_id_stats(lidar_ids: np.ndarray) -> tuple[str, list[tuple[int, int, float]]]:
    """Return (summary text, list of (id, count, pct))."""
    ids, counts = np.unique(np.asarray(lidar_ids).astype(np.int32), return_counts=True)
    total = int(counts.sum()) or 1
    rows = [(int(i), int(c), 100.0 * float(c) / total) for i, c in zip(ids, counts)]
    id_list = [r[0] for r in rows]
    parts = [f"id={i}:{c}({p:.1f}%)" for i, c, p in rows]
    text = (
        f"lidar_id field → unique={id_list}  n_lidars={len(id_list)}  |  "
        + "  ".join(parts)
        + f"  |  N={total}"
    )
    return text, rows


def render_lidar_id_banner(
    lidar_ids: np.ndarray,
    width: int,
    height: int = 72,
) -> np.ndarray:
    """Full-width strip listing lidar_id values/counts with color swatches."""
    text, rows = format_lidar_id_stats(lidar_ids)
    banner = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.putText(
        banner,
        text[:220],
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (240, 240, 240),
        2,
    )
    x = 16
    y = 58
    for lid, count, pct in rows:
        bgr = LIDAR_ID_BGR.get(lid, LIDAR_ID_DEFAULT_BGR)
        cv2.rectangle(banner, (x, y - 16), (x + 28, y + 4), bgr, -1)
        cv2.rectangle(banner, (x, y - 16), (x + 28, y + 4), (255, 255, 255), 1)
        label = f"lidar_id={lid}  n={count}  {pct:.1f}%"
        cv2.putText(
            banner,
            label,
            (x + 36, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
        )
        x += 420
    return banner


def _match_width(img: np.ndarray, grid_w: int) -> np.ndarray:
    if img.shape[1] == grid_w:
        return img
    scale = grid_w / float(img.shape[1])
    nh = max(1, int(round(img.shape[0] * scale)))
    return cv2.resize(img, (grid_w, nh), interpolation=cv2.INTER_AREA)


def compose_frame(
    cam_panels: list[tuple[str, np.ndarray]],
    lidar_panels: list[np.ndarray],
    title: str,
    tile_w: int = 1280,
    tile_h: int = 720,
    lidar_id_banner: np.ndarray | None = None,
) -> np.ndarray:
    """2x5 camera tiles + optional lidar_id banner + stacked lidar panels."""
    tiles = []
    for name, img in cam_panels:
        canvas = fit_into(img, tile_w, tile_h)
        cv2.putText(canvas, name, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        tiles.append(canvas)

    while len(tiles) < 10:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    row1 = np.hstack(tiles[:5])
    row2 = np.hstack(tiles[5:10])
    grid_w = row1.shape[1]

    panels = [_match_width(p, grid_w) for p in lidar_panels]
    chunks = [row1, row2]
    if lidar_id_banner is not None:
        chunks.append(_match_width(lidar_id_banner, grid_w))
    chunks.extend(panels)
    body = np.vstack(chunks)
    header = np.zeros((56, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title[:200], (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (240, 240, 240), 2)
    return np.vstack([header, body])


def list_clip_frames(clip_dir: Path) -> list[str]:
    idx_path = clip_dir / "frames_index.json"
    if idx_path.is_file():
        entries = json.loads(idx_path.read_text())
        return [str(e["timestamp"]) for e in entries if e.get("has_lidar")]
    frames = []
    for p in sorted((clip_dir / "frames").iterdir()):
        if (p / "lidar_merge.bin").is_file():
            frames.append(p.name)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clip",
        default="stop_1784423032302844849_vehicle-V002-20260719_090818",
    )
    ap.add_argument("--backup-root", default="data/robotruck_clips_backup")
    ap.add_argument("--out-dir", default="exp/robotruck/clip_video")
    ap.add_argument("--config-file", default="configs/waymo/semseg-litept-small-v1m1.py")
    ap.add_argument(
        "--weight",
        default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth",
    )
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all after stride")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse-pred", action="store_true", help="Reuse cached pred npy if present")
    ap.add_argument("--save-frame-jpgs", action="store_true")
    ap.add_argument("--tile-w", type=int, default=1280, help="Camera panel width")
    ap.add_argument("--tile-h", type=int, default=720, help="Camera panel height")
    ap.add_argument(
        "--bev-y-min",
        type=float,
        default=-200.0,
        help="BEV forward (+y) min meters",
    )
    ap.add_argument(
        "--bev-y-max",
        type=float,
        default=400.0,
        help="BEV forward (+y) max meters",
    )
    ap.add_argument(
        "--bev-x-half",
        type=float,
        default=30.0,
        help="BEV lateral (±x) half-width meters (tight to reduce empty black)",
    )
    ap.add_argument(
        "--z-min",
        type=float,
        default=-5.0,
        help="Side-view height min (meters)",
    )
    ap.add_argument(
        "--z-max",
        type=float,
        default=20.0,
        help="Side-view height max (meters)",
    )
    ap.add_argument("--proj-radius", type=int, default=3, help="Projection point radius on tile")
    ap.add_argument("--out-name", default="", help="Optional video filename stem suffix")
    ap.add_argument(
        "--aggregate-static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip-level static aggregation via ego_pose (default on)",
    )
    ap.add_argument("--static-voxel", type=float, default=0.2, help="Static agg voxel size (m)")
    ap.add_argument(
        "--agg-stride",
        type=int,
        default=2,
        help="Stride over clip frames when building static aggregate",
    )
    ap.add_argument(
        "--rebuild-static-agg",
        action="store_true",
        help="Ignore cached static aggregate and rebuild",
    )
    ap.add_argument(
        "--occupancy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add voxel occupancy panels (default on on dev_occ)",
    )
    ap.add_argument(
        "--occ-voxel",
        type=float,
        default=0.4,
        help="Occupancy voxel size in meters",
    )
    ap.add_argument(
        "--occ-min-points",
        type=int,
        default=1,
        help="Min points in a voxel to mark occupied",
    )
    args = ap.parse_args()

    clip_dir = (ROOT / args.backup_root / args.clip).resolve()
    if not clip_dir.is_dir():
        raise FileNotFoundError(clip_dir)
    out_root = (ROOT / args.out_dir / args.clip).resolve()
    pred_dir = out_root / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir = out_root / "frames_jpg_occ"
    if args.save_frame_jpgs:
        jpg_dir.mkdir(parents=True, exist_ok=True)

    all_ts = list_clip_frames(clip_dir)
    timestamps = all_ts[:: max(1, args.stride)]
    if args.max_frames > 0:
        timestamps = timestamps[: args.max_frames]
    print(f"clip={args.clip} frames={len(timestamps)} stride={args.stride}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = _h.load_segmentor(ROOT / args.config_file, ROOT / args.weight, device)

    bev_target_w = args.tile_w * 5
    x_range = (-args.bev_x_half, args.bev_x_half)
    y_range = (args.bev_y_min, args.bev_y_max)
    z_range = (args.z_min, args.z_max)
    lidar_legend = [(f"lidar{k}", v) for k, v in LIDAR_ID_BGR.items()]
    suffix = args.out_name or f"stride{args.stride}_occ"
    video_path = out_root / f"{args.clip}_{suffix}.mp4"
    writer = None

    static_agg = None
    if args.aggregate_static:
        cache_path = out_root / "static_agg" / f"static_voxel{args.static_voxel:g}_s{args.agg_stride}.npz"
        if args.rebuild_static_agg and cache_path.is_file():
            cache_path.unlink()
        agg_ts = all_ts[:: max(1, args.agg_stride)]
        print(
            f"building/loading static aggregate n_src={len(agg_ts)} voxel={args.static_voxel} "
            f"(oracle boxes exclude movers; no point-level track ids)",
            flush=True,
        )
        static_agg = sag.load_or_build_static_aggregate(
            clip_dir,
            pred_dir,
            agg_ts,
            load_lidar_bin=_h.load_lidar_bin,
            lidar_cols=len(_h.LIDAR_COLS),
            infer_frame=_h.infer_frame if not args.reuse_pred else None,
            model=model,
            device=device,
            grid_size=args.grid_size,
            voxel=args.static_voxel,
            cache_path=cache_path,
            use_oracle_boxes=True,
        )
        # If cache miss and reuse-pred skipped infer, retry allowing infer for missing preds
        if static_agg["xyz_map"].shape[0] == 0 and args.reuse_pred:
            static_agg = sag.load_or_build_static_aggregate(
                clip_dir,
                pred_dir,
                agg_ts,
                load_lidar_bin=_h.load_lidar_bin,
                lidar_cols=len(_h.LIDAR_COLS),
                infer_frame=_h.infer_frame,
                model=model,
                device=device,
                grid_size=args.grid_size,
                voxel=args.static_voxel,
                cache_path=cache_path,
                use_oracle_boxes=True,
            )
        print(
            f"static_agg points={static_agg['xyz_map'].shape[0]} "
            f"frames={static_agg['n_frames']} cache={static_agg.get('from_cache')}",
            flush=True,
        )

    for i, ts in enumerate(timestamps):
        fr = clip_dir / "frames" / ts
        meta = json.loads((fr / "frame.json").read_text())
        sensors = meta["dependency"]["sensors"]

        pts = _h.load_lidar_bin(fr / "lidar_merge.bin", num_cols=len(_h.LIDAR_COLS))
        coord = pts[:, :3].astype(np.float32)
        lidar_ids = pts[:, 6].astype(np.int32)
        intensity = pts[:, 3]
        strength = np.tanh(intensity.reshape(-1, 1) / 255.0).astype(np.float32)

        pred_path = pred_dir / f"{ts}_pred.npy"
        if args.reuse_pred and pred_path.is_file():
            pred = np.load(pred_path).astype(np.int64).reshape(-1)
            if pred.shape[0] != coord.shape[0]:
                pred = _h.infer_frame(model, coord, strength, device, args.grid_size)
                np.save(pred_path, pred.astype(np.int32))
        else:
            pred = _h.infer_frame(model, coord, strength, device, args.grid_size)
            np.save(pred_path, pred.astype(np.int32))

        # Clip-static (map→vehicle) + frame-dynamic
        lab_s = np.zeros((0,), np.int32)
        lid_s = np.zeros((0,), np.int32)
        xyz_s = np.zeros((0, 3), np.float32)
        if static_agg is not None and static_agg["xyz_map"].shape[0] > 0:
            pose = (meta.get("dependency") or {}).get("ego_pose", {}).get("pose")
            if pose:
                T_map_v = sag.ego_pose_to_T_map_vehicle(pose)
                xyz_s, lab_s, lid_s = sag.static_in_vehicle(
                    static_agg,
                    T_map_v,
                    x_range=(x_range[0] * 1.5, x_range[1] * 1.5),
                    y_range=y_range,
                    z_range=(z_range[0] - 2.0, z_range[1] + 5.0),
                )
                vis_xyz, vis_lab, vis_lid, is_dyn = sag.merge_static_dynamic(
                    xyz_s, lab_s, lid_s, coord, pred, lidar_ids
                )
            else:
                vis_xyz, vis_lab, vis_lid = coord, pred, lidar_ids
                is_dyn = np.isin(pred.astype(np.int64), list(sag.WAYMO_DYNAMIC_IDS))
        else:
            vis_xyz, vis_lab, vis_lid = coord, pred, lidar_ids
            is_dyn = np.isin(pred.astype(np.int64), list(sag.WAYMO_DYNAMIC_IDS))

        cam_panels: list[tuple[str, np.ndarray]] = []
        for cam_name in CAM_ORDER:
            img_path = fr / f"{cam_name}.jpg"
            if not img_path.is_file() or cam_name not in sensors:
                continue
            cam_doc = sensors[cam_name]
            K, dist5, T_c_v, cal_w, cal_h = parse_camera(cam_doc)
            img = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
            ih, iw = img.shape[:2]
            if iw != cal_w or ih != cal_h:
                sx, sy = iw / float(cal_w), ih / float(cal_h)
                K = K.copy()
                K[0, :] *= sx
                K[1, :] *= sy

            tile = fit_into(img, args.tile_w, args.tile_h)
            scale = min(args.tile_w / iw, args.tile_h / ih, 1.0)
            nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
            x0 = (args.tile_w - nw) // 2
            y0 = (args.tile_h - nh) // 2
            K_s = K.copy()
            K_s[0, :] *= scale
            K_s[1, :] *= scale
            uv, cols = project_points(
                vis_xyz, vis_lab, K_s, dist5, T_c_v, nw, nh, max_points=100000, seed=i
            )
            if uv.shape[0]:
                uv = uv.copy()
                uv[:, 0] += x0
                uv[:, 1] += y0
            proj = draw_projection(
                tile, uv, cols, radius=args.proj_radius, alpha=0.88
            )
            cam_panels.append((cam_name, proj))

        seg_cols = labels_to_bgr(vis_lab)
        lid_cols = lidar_ids_to_bgr(vis_lid)
        lid_stats_text, lid_rows = format_lidar_id_stats(lidar_ids)
        n_static = int((~is_dyn).sum()) if is_dyn.shape[0] == vis_xyz.shape[0] else xyz_s.shape[0]
        n_dyn = int(is_dyn.sum()) if is_dyn.shape[0] else 0
        print(
            f"  ts={ts} {lid_stats_text} | vis N={vis_xyz.shape[0]} "
            f"static_agg_in_roi={xyz_s.shape[0]} dyn_frame={n_dyn}",
            flush=True,
        )

        bev_seg = render_bev_bgr(
            vis_xyz,
            seg_cols,
            x_range=x_range,
            y_range=y_range,
            target_w=bev_target_w,
            title=(
                f"BEV seg combined (static clip-agg + dyn frame)  "
                f"N={vis_xyz.shape[0]} static_roi={xyz_s.shape[0]} dyn={n_dyn}"
            ),
            seed=i,
        )
        bev_static = render_bev_bgr(
            xyz_s,
            labels_to_bgr(lab_s) if xyz_s.shape[0] else np.zeros((0, 3), np.uint8),
            x_range=x_range,
            y_range=y_range,
            target_w=bev_target_w,
            title=f"BEV static-only clip-agg  N={xyz_s.shape[0]} voxel={args.static_voxel:g}m",
            seed=i,
        )
        bev_lid = render_bev_bgr(
            vis_xyz,
            lid_cols,
            x_range=x_range,
            y_range=y_range,
            target_w=bev_target_w,
            title=(
                f"BEV lidar_id (combined) unique={[r[0] for r in lid_rows]}  "
                f"(+y {int(y_range[0])}..{int(y_range[1])}m)"
            ),
            legend_items=lidar_legend,
            seed=i,
        )
        side_yz_seg, side_xz_seg = render_side_views(
            vis_xyz,
            seg_cols,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            target_w=bev_target_w,
            seed=i,
            label="seg combined",
        )
        side_yz_lid, side_xz_lid = render_side_views(
            vis_xyz,
            lid_cols,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            target_w=bev_target_w,
            seed=i,
            legend_items=lidar_legend,
            label="lidar_id",
        )

        lidar_panels = [
            bev_seg,
            bev_static,
            bev_lid,
            side_yz_seg,
            side_xz_seg,
            side_yz_lid,
            side_xz_lid,
        ]

        if args.occupancy:
            grid = occmod.build_occupancy(
                vis_xyz,
                vis_lab,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                voxel=args.occ_voxel,
                min_points=args.occ_min_points,
            )
            col_sem = occmod.occ_semantic_colors(grid, labels_to_bgr)
            col_h = occmod.occ_height_colors(grid)
            col_bin = occmod.occ_binary_colors(grid)
            occ_bev_sem = occmod.render_occ_bev(
                grid,
                colors_bgr=col_sem,
                target_w=bev_target_w,
                title="Occ BEV semantic (voxel occupied)",
            )
            occ_bev_h = occmod.render_occ_bev(
                grid,
                colors_bgr=col_h,
                target_w=bev_target_w,
                title="Occ BEV height (max z in column)",
            )
            occ_bev_bin = occmod.render_occ_bev(
                grid,
                colors_bgr=col_bin,
                target_w=bev_target_w,
                title="Occ BEV binary occupied",
            )
            occ_side = occmod.render_occ_side_yz(
                grid,
                colors_bgr=col_sem,
                target_w=bev_target_w,
                title="Occ Side YZ semantic",
            )
            lidar_panels.extend([occ_bev_sem, occ_bev_h, occ_bev_bin, occ_side])
            print(
                f"    occ voxels={grid.centers.shape[0]} shape={grid.shape} "
                f"voxel={grid.voxel:g}m",
                flush=True,
            )

        id_banner = render_lidar_id_banner(lidar_ids, bev_target_w)
        # Append static-agg note on banner
        note = (
            f"static_agg: N_map={0 if static_agg is None else static_agg['xyz_map'].shape[0]}  "
            f"roi={xyz_s.shape[0]}  dyn_frame={n_dyn}  "
            f"occ_voxel={args.occ_voxel:g}m"
        )
        cv2.putText(
            id_banner,
            note[:180],
            (16, id_banner.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1,
        )
        title = (
            f"{args.clip}  ts={ts}  [{i+1}/{len(timestamps)}]  |  "
            f"lidar_id={[r[0] for r in lid_rows]}  static_roi={xyz_s.shape[0]} dyn={n_dyn}"
        )
        frame = compose_frame(
            cam_panels,
            lidar_panels,
            title,
            tile_w=args.tile_w,
            tile_h=args.tile_h,
            lidar_id_banner=id_banner,
        )

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(video_path), fourcc, args.fps, (frame.shape[1], frame.shape[0])
            )
            print(f"video {frame.shape[1]}x{frame.shape[0]} -> {video_path}")
        writer.write(frame)
        if args.save_frame_jpgs:
            cv2.imwrite(
                str(jpg_dir / f"{i:05d}_{ts}.jpg"),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(timestamps)}] done", flush=True)

    if writer is not None:
        writer.release()
    print(f"done -> {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
