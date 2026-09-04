"""Debug: MP4 from an existing OCC scene package (no re-inference)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_VIEW = Path(__file__).resolve().parent
if str(_VIEW) not in sys.path:
    sys.path.insert(0, str(_VIEW))

from env_paths import ensure_c_path

ensure_c_path()
import occ_render as occmod


def _read_f32(path: Path, n3: bool = True) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    return arr.reshape(-1, 3) if n3 else arr


def _read_u8(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


def _read_i32(path: Path, n3: bool = False) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.int32)
    return arr.reshape(-1, 3) if n3 else arr


def labels_to_bgr(labels: np.ndarray, colors_rgb: list) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = len(colors_rgb)
    out = np.zeros((lab.shape[0], 3), dtype=np.uint8)
    valid = (lab >= 0) & (lab < n)
    idx = lab[valid]
    rgb = np.asarray([colors_rgb[i] for i in idx], dtype=np.float64)
    if rgb.size and float(np.nanmax(rgb)) <= 1.5:
        rgb = rgb * 255.0
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    out[valid] = rgb[:, ::-1]  # RGB → BGR
    return out


def project_centers(
    xyz: np.ndarray,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    width: int,
    height: int,
    max_n: int = 180000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (uv Nx2, z N, keep_idx) in camera image."""
    n = xyz.shape[0]
    if n == 0:
        return (
            np.zeros((0, 2), np.float64),
            np.zeros((0,), np.float64),
            np.zeros((0,), np.int64),
        )
    idx = np.arange(n)
    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        xyz = xyz[idx]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    ph = np.hstack([xyz.astype(np.float64), ones])
    pc = (T_c_v @ ph.T).T[:, :3]
    front = pc[:, 2] > 0.3
    pc = pc[front]
    idx = idx[front]
    if pc.shape[0] == 0:
        return (
            np.zeros((0, 2), np.float64),
            np.zeros((0,), np.float64),
            np.zeros((0,), np.int64),
        )
    uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist5)
    uv = uv.reshape(-1, 2)
    z = pc[:, 2]
    inside = (
        (uv[:, 0] >= -40)
        & (uv[:, 0] < width + 40)
        & (uv[:, 1] >= -40)
        & (uv[:, 1] < height + 40)
        & np.isfinite(uv).all(axis=1)
    )
    return uv[inside], z[inside], idx[inside]


def _project_corners_batch(
    corners_xyz: np.ndarray,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    *,
    near_z: float = 1.0,
    max_r2: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N,8,3) vehicle-frame corners → uv (N,8,2), z (N,8), valid (N,8).

    max_r2 only guards radtan blow-ups (not a circular image mask). Use a
    generous default so wide pinhole cams keep full rectangular FOV.
    """
    n = corners_xyz.shape[0]
    flat = corners_xyz.reshape(-1, 3)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    hv = np.hstack([flat.astype(np.float64), ones])
    pc = (T_c_v @ hv.T).T[:, :3]
    z = pc[:, 2]
    xn = np.zeros(flat.shape[0], dtype=np.float64)
    yn = np.zeros_like(xn)
    valid = z > near_z
    np.divide(pc[:, 0], z, out=xn, where=valid)
    np.divide(pc[:, 1], z, out=yn, where=valid)
    r2 = xn * xn + yn * yn
    valid &= r2 <= max_r2
    k1, k2, p1, p2, k3 = [float(dist5[i]) if i < len(dist5) else 0.0 for i in range(5)]
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    valid &= np.isfinite(radial) & (np.abs(radial) <= 20)
    xpp = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    ypp = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u = fx * xpp + cx
    v = fy * ypp + cy
    uv = np.stack([u, v], axis=1).reshape(n, 8, 2)
    zz = z.reshape(n, 8)
    ok = valid.reshape(n, 8) & np.isfinite(uv).all(axis=-1)
    return uv, zz, ok


def overlay_occ_squares(
    image_bgr: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    colors_rgb: list,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    voxel: float,
    alpha: float = 0.55,
    max_n: int = 60000,
    ijk: np.ndarray | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Project each occupied voxel cube (8 corners → convex hull) onto the image.

    Matches tools/occ_viewer/app.js projectOccCubes (cull behind-cam / wings /
    oversized silhouettes; rectangular FOV only — no circular r² mask).
    """
    if ijk is None or x_range is None or y_range is None or z_range is None:
        return image_bgr

    v = float(voxel)
    x0, y0, z0 = float(x_range[0]), float(y_range[0]), float(z_range[0])
    ijk = np.asarray(ijk, dtype=np.int32).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1)
    n = ijk.shape[0]
    if n == 0:
        return image_bgr
    # Same as viewer: deterministic stride, not random subsample.
    if n > max_n:
        step = int(np.ceil(n / max_n))
        ijk = ijk[::step]
        labels = labels[::step]
        n = ijk.shape[0]

    ix = ijk[:, 0].astype(np.float64)
    iy = ijk[:, 1].astype(np.float64)
    iz = ijk[:, 2].astype(np.float64)
    xa, xb = x0 + ix * v, x0 + (ix + 1) * v
    ya, yb = y0 + iy * v, y0 + (iy + 1) * v
    za, zb = z0 + iz * v, z0 + (iz + 1) * v
    # (N,8,3) cube corners
    corners = np.stack(
        [
            np.stack([xa, ya, za], axis=1),
            np.stack([xb, ya, za], axis=1),
            np.stack([xb, yb, za], axis=1),
            np.stack([xa, yb, za], axis=1),
            np.stack([xa, ya, zb], axis=1),
            np.stack([xb, ya, zb], axis=1),
            np.stack([xb, yb, zb], axis=1),
            np.stack([xa, yb, zb], axis=1),
        ],
        axis=1,
    )
    uv, zz, ok = _project_corners_batch(corners, K, dist5, T_c_v)
    h, w = image_bgr.shape[:2]
    u_lo, u_hi = -0.02 * w, 1.02 * w
    v_lo, v_hi = -0.02 * h, 1.02 * h
    max_span = 0.12 * max(w, h)
    # Voxel centers must project inside the image (avoids wing artifacts).
    ctr = np.stack(
        [0.5 * (xa + xb), 0.5 * (ya + yb), 0.5 * (za + zb)], axis=1
    )
    # reuse 8-corner batch by repeating center 8x (wasteful but simple/correct)
    ctr8 = np.repeat(ctr[:, None, :], 8, axis=1)
    uv_c, zz_c, ok_c = _project_corners_batch(ctr8, K, dist5, T_c_v)
    uv_c = uv_c[:, 0, :]
    ok_c = ok_c[:, 0]
    center_in = (
        ok_c
        & (uv_c[:, 0] >= 0)
        & (uv_c[:, 0] < w)
        & (uv_c[:, 1] >= 0)
        & (uv_c[:, 1] < h)
    )
    in_fov = (
        (uv[:, :, 0] >= u_lo)
        & (uv[:, :, 0] <= u_hi)
        & (uv[:, :, 1] >= v_lo)
        & (uv[:, :, 1] <= v_hi)
    )
    span_u = uv[:, :, 0].max(axis=1) - uv[:, :, 0].min(axis=1)
    span_v = uv[:, :, 1].max(axis=1) - uv[:, :, 1].min(axis=1)
    span_ok = (span_u <= max_span) & (span_v <= max_span)
    zmean = zz.mean(axis=1)
    expect = (2.5 * v * float(K[0, 0])) / np.maximum(zmean, 1.0)
    size_ok = (span_u <= np.maximum(24.0, 3.0 * expect)) & (
        span_v <= np.maximum(24.0, 3.0 * expect)
    )
    keep = ok.all(axis=1) & in_fov.all(axis=1) & span_ok & size_ok & center_in
    cols = labels_to_bgr(labels, colors_rgb)
    overlay = image_bgr.copy()
    items = []
    for i in np.where(keep)[0]:
        pts = uv[i].astype(np.float32)
        zmean = float(zz[i].mean())
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        if hull is None or len(hull) < 3:
            continue
        items.append((zmean, hull, cols[i]))
    items.sort(key=lambda t: -t[0])
    for _, hull, col in items:
        poly = np.rint(hull.reshape(-1, 2)).astype(np.int32)
        if poly.shape[0] < 3:
            continue
        cv2.fillConvexPoly(overlay, poly, (int(col[0]), int(col[1]), int(col[2])))
    return cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)


def overlay_points_pixels(
    image_bgr: np.ndarray,
    xyz: np.ndarray,
    labels: np.ndarray,
    colors_rgb: list,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    alpha: float = 0.85,
    max_n: int = 150000,
) -> np.ndarray:
    """Draw point cloud as single image pixels (true point projection)."""
    h, w = image_bgr.shape[:2]
    uv, z, keep = project_centers(xyz, K, dist5, T_c_v, w, h, max_n=max_n)
    if uv.shape[0] == 0:
        return image_bgr
    cols = labels_to_bgr(labels[keep], colors_rgb)
    order = np.argsort(-z)
    uv, cols = uv[order], cols[order]
    overlay = image_bgr.copy()
    uu = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, w - 1)
    vv = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, h - 1)
    overlay[vv, uu] = cols
    return cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)


def load_frame_occ(frame_dir: Path, meta: dict) -> occmod.OccupancyGrid:
    if meta.get("grid"):
        voxel = float(meta["grid"]["voxel"])
        x_range = tuple(meta["grid"]["x_range"])
        y_range = tuple(meta["grid"]["y_range"])
        z_range = tuple(meta["grid"]["z_range"])
        shape = tuple(meta["grid"].get("shape") or [1, 1, 1])
    else:
        voxel = float(meta["voxel"])
        x_range = tuple(meta["x_range"])
        y_range = tuple(meta["y_range"])
        z_range = tuple(meta["z_range"])
        shape = tuple(meta.get("occ_shape") or [1, 1, 1])

    if meta.get("assets") and meta["assets"].get("occupancy"):
        occ = meta["assets"]["occupancy"]
        # uri is scene-root relative; frame_dir is .../frames/ts
        scene_root = frame_dir.parent.parent

        def _uri(key):
            u = occ[key]["uri"]
            return scene_root / u

        centers = _read_f32(_uri("centers"), True)
        labels = _read_u8(_uri("labels")).astype(np.int32)
        ijk = _read_i32(_uri("ijk"), True)
        counts = _read_i32(_uri("counts"), False)
    else:
        occ = meta["occupancy"]
        centers = _read_f32(frame_dir / occ["centers"], True)
        labels = _read_u8(frame_dir / occ["labels"]).astype(np.int32)
        ijk = _read_i32(frame_dir / occ["ijk"], True)
        counts = _read_i32(frame_dir / occ["counts"], False)
    max_z = centers[:, 2].astype(np.float32) if centers.size else np.zeros((0,), np.float32)
    return occmod.OccupancyGrid(
        ijk=ijk,
        centers=centers,
        labels=labels,
        counts=counts,
        max_z=max_z,
        voxel=voxel,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        shape=shape,
    )


def fit_panel(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def compose_layout(
    cam_panels: list[np.ndarray],
    bev: np.ndarray,
    title: str,
    tile_w: int,
    tile_h: int,
    cols: int = 5,
) -> np.ndarray:
    n = len(cam_panels)
    rows = int(np.ceil(n / cols)) if n else 0
    grid_h = rows * tile_h
    grid_w = cols * tile_w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, p in enumerate(cam_panels):
        r, c = divmod(i, cols)
        grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = p

    bev_h = min(480, max(200, bev.shape[0]))
    bev_fit = fit_panel(bev, grid_w, bev_h)
    header = np.full((40, grid_w, 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        title[:180],
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
    )
    return np.vstack([header, grid, bev_fit])


def draw_legend_strip(colors_rgb: list, names: list[str], width: int) -> np.ndarray:
    n = len(colors_rgb)
    h = 36
    img = np.full((h, width, 3), 20, dtype=np.uint8)
    if n == 0:
        return img
    cell = max(1, width // n)
    for i, rgb in enumerate(colors_rgb):
        x0 = i * cell
        x1 = min(width, x0 + cell - 2)
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(img, (x0, 4), (x1, h - 4), bgr, -1)
        name = names[i] if i < len(names) else str(i)
        cv2.putText(
            img,
            name[:10],
            (x0 + 2, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
    return img


def export_scene_video(
    scene_dir: Path,
    *,
    out_path: Path | None = None,
    mode: str = "occ",
    fps: float = 5.0,
    tile_w: int = 960,
    tile_h: int = 540,
    max_frames: int = 0,
    status_cb=None,
) -> Path:
    scene_dir = scene_dir.resolve()
    index = json.loads((scene_dir / "index.json").read_text())
    frames = list(index.get("frames") or [])
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise SystemExit(f"no frames in {scene_dir}/index.json")

    videos_dir = scene_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    clip = index.get("clip") or scene_dir.name
    # proj=hull2: cube corners → convex hull (viewer-parity). Old exports used
    # axis-aligned squares from voxel centers — do not overwrite those names.
    stem = f"{clip}_scene_{mode}_v{index.get('occ_voxel', 'x')}_hull2"
    if out_path is None:
        out_path = videos_dir / f"{stem}.mp4"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    t0 = time.time()
    for i, fr in enumerate(frames):
        frame_dir = scene_dir / fr["dir"]
        meta = json.loads((frame_dir / "meta.json").read_text())
        tax = (index.get("taxonomy") or {}).get("fine") or {}
        colors_rgb = (
            tax.get("colors_rgb")
            or meta.get("class_colors_rgb")
            or []
        )
        names = tax.get("names") or meta.get("class_names") or []
        voxel = float((meta.get("grid") or {}).get("voxel") or meta.get("voxel") or 0.2)
        grid = load_frame_occ(frame_dir, meta)
        colors_bgr = labels_to_bgr(grid.labels, colors_rgb)

        pts_xyz = pts_lab = None
        if meta.get("points") and mode in ("points", "both"):
            pi = meta["points"]
            pts_xyz = _read_f32(frame_dir / pi["xyz"], True)
            pts_lab = _read_u8(frame_dir / pi["labels"])

        cam_panels = []
        for cam in meta.get("cameras") or []:
            img_path = frame_dir / cam["file"]
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((cam["height"], cam["width"], 3), dtype=np.uint8)
            K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
            dist5 = np.asarray(cam.get("dist5") or [0, 0, 0, 0, 0], dtype=np.float64)
            T_c_v = np.asarray(cam["T_c_v"], dtype=np.float64).reshape(4, 4)
            if mode in ("occ", "both"):
                img = overlay_occ_squares(
                    img,
                    grid.centers,
                    grid.labels,
                    colors_rgb,
                    K,
                    dist5,
                    T_c_v,
                    voxel,
                    ijk=grid.ijk,
                    x_range=grid.x_range,
                    y_range=grid.y_range,
                    z_range=grid.z_range,
                )
            if mode in ("points", "both") and pts_xyz is not None:
                img = overlay_points_pixels(
                    img, pts_xyz, pts_lab, colors_rgb, K, dist5, T_c_v
                )
            # name badge
            cv2.putText(
                img,
                cam["name"],
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (240, 240, 240),
                2,
            )
            cam_panels.append(fit_panel(img, tile_w, tile_h))

        bev = occ_render.render_occ_bev(
            grid,
            colors_bgr=colors_bgr,
            target_w=max(tile_w * 5, 2400),
            title=f"Occ BEV  voxel={voxel:g}m",
        )
        title = (
            f"{clip}  ts={meta['timestamp']}  [{i+1}/{len(frames)}]  "
            f"mode={mode}  n_occ={meta['n_occ']}  voxel={voxel:g}m"
        )
        frame = compose_layout(cam_panels, bev, title, tile_w, tile_h, cols=5)
        legend = draw_legend_strip(colors_rgb, names, frame.shape[1])
        frame = np.vstack([frame, legend])

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_path), fourcc, float(fps), (frame.shape[1], frame.shape[0])
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to open VideoWriter for {out_path}")
            msg = f"writing {frame.shape[1]}x{frame.shape[0]} @ {fps}fps -> {out_path}"
            print(msg, flush=True)
            if status_cb:
                status_cb({"state": "running", "message": msg, "frame": i, "n": len(frames)})

        writer.write(frame)
        if status_cb:
            status_cb(
                {
                    "state": "running",
                    "message": f"frame {i+1}/{len(frames)}",
                    "frame": i + 1,
                    "n": len(frames),
                    "path": str(out_path),
                }
            )
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(frames)}] done", flush=True)

    if writer is not None:
        writer.release()
    elapsed = time.time() - t0
    meta_out = {
        "clip": clip,
        "mode": mode,
        "fps": fps,
        "n_frames": len(frames),
        "path": str(out_path),
        "relpath": str(out_path.relative_to(scene_dir)),
        "elapsed_sec": round(elapsed, 2),
        "occ_voxel": index.get("occ_voxel"),
    }
    (videos_dir / f"{out_path.stem}_meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"done -> {out_path} ({elapsed:.1f}s)", flush=True)
    if status_cb:
        status_cb({"state": "done", "message": "ok", "path": str(out_path), **meta_out})
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="Exported scene root (index.json)")
    ap.add_argument("--out", default="", help="Output mp4 path (default under scene/videos/)")
    ap.add_argument(
        "--mode",
        choices=["none", "occ", "points", "both"],
        default="occ",
        help="Camera overlay mode",
    )
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--tile-w", type=int, default=960)
    ap.add_argument("--tile-h", type=int, default=540)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all scene frames")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    export_scene_video(
        Path(args.scene),
        out_path=out,
        mode=args.mode,
        fps=args.fps,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
