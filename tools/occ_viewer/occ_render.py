"""OCC render helpers for scene_video / clip video (C lane; not used by export)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OccupancyGrid:
    ijk: np.ndarray
    centers: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    max_z: np.ndarray
    voxel: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    shape: tuple[int, int, int]


def _height_to_bgr(z: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Map height to a blue→cyan→yellow→red ramp (BGR)."""
    t = np.clip((z - z0) / max(1e-6, z1 - z0), 0.0, 1.0)
    # piecewise RGB then → BGR
    r = np.clip(1.5 * t - 0.25, 0, 1)
    g = np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0, 1)
    b = np.clip(1.25 - 1.5 * t, 0, 1)
    rgb = np.stack([r, g, b], axis=1)
    return (rgb * 255.0).astype(np.uint8)[:, ::-1]


def render_occ_bev(
    occ: OccupancyGrid,
    *,
    colors_bgr: np.ndarray,
    target_w: int,
    title: str = "Occupancy BEV",
    collapse: str = "any",  # any | max_z_cell
) -> np.ndarray:
    """Draw occupied voxels as filled squares in landscape BEV (+y→, +x↓)."""
    y0, y1 = occ.y_range
    x0, x1 = occ.x_range
    fwd_span = max(1e-6, y1 - y0)
    lat_span = max(1e-6, x1 - x0)
    ppm = float(target_w) / fwd_span
    out_w = int(target_w)
    out_h = max(1, int(round(lat_span * ppm)))
    img = np.full((out_h, out_w, 3), 18, dtype=np.uint8)

    if occ.centers.shape[0] == 0:
        cv2.putText(
            img,
            f"{title}  empty  voxel={occ.voxel:g}m",
            (12, out_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2,
        )
        return img

    # Collapse Z: one cell per (ix,iy) — keep max-count among z-stack
    ix = occ.ijk[:, 0]
    iy = occ.ijk[:, 1]
    flat = ix.astype(np.int64) + occ.shape[0] * iy.astype(np.int64)
    order2 = np.lexsort((-occ.counts, flat))
    flat2 = flat[order2]
    uniq, start2 = np.unique(flat2, return_index=True)
    best = order2[start2]

    # Exact abutting cells from ijk (solid grid, not sparse center dots)
    cell = max(1, int(round(occ.voxel * ppm)))
    iy_b = occ.ijk[best, 1].astype(np.int32)
    ix_b = occ.ijk[best, 0].astype(np.int32)
    cols = colors_bgr[best]
    u0 = np.floor(iy_b.astype(np.float64) * occ.voxel * ppm).astype(np.int32)
    v0 = np.floor(ix_b.astype(np.float64) * occ.voxel * ppm).astype(np.int32)
    if cell <= 1:
        m = (u0 >= 0) & (u0 < out_w) & (v0 >= 0) & (v0 < out_h)
        img[v0[m], u0[m]] = cols[m]
    else:
        for u, v, c in zip(u0, v0, cols):
            x1 = min(out_w, int(u) + cell)
            y1 = min(out_h, int(v) + cell)
            if x1 <= 0 or y1 <= 0 or u >= out_w or v >= out_h:
                continue
            cv2.rectangle(
                img,
                (max(0, int(u)), max(0, int(v))),
                (x1 - 1, y1 - 1),
                (int(c[0]), int(c[1]), int(c[2])),
                -1,
            )

    # distance guides
    for d in (-200, -100, 0, 100, 200, 300, 400):
        if y0 <= d <= y1:
            uu = int(round((d - y0) * ppm))
            cv2.line(img, (uu, 0), (uu, out_h - 1), (0, 180, 220), 1)
            cv2.putText(
                img,
                f"{d:g}m",
                (uu + 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 220),
                1,
            )

    n_xy = int(uniq.size)
    cv2.putText(
        img,
        f"{title}  voxels_xy={n_xy}/{occ.centers.shape[0]}  voxel={occ.voxel:g}m  {ppm:.2f}px/m",
        (12, out_h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
    )
    return img


def render_occ_side_yz(
    occ: OccupancyGrid,
    *,
    colors_bgr: np.ndarray,
    target_w: int,
    title: str = "Occupancy Side YZ",
) -> np.ndarray:
    """Occupied voxels collapsed over lateral x → YZ plane (+y→, +z↑)."""
    y0, y1 = occ.y_range
    z0, z1 = occ.z_range
    fwd_span = max(1e-6, y1 - y0)
    z_span = max(1e-6, z1 - z0)
    ppm = float(target_w) / fwd_span
    out_w = int(target_w)
    out_h = max(1, int(round(z_span * ppm)))
    img = np.full((out_h, out_w, 3), 18, dtype=np.uint8)

    if occ.centers.shape[0] == 0:
        cv2.putText(
            img,
            f"{title}  empty",
            (12, max(24, out_h - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2,
        )
        return img

    iy = occ.ijk[:, 1]
    iz = occ.ijk[:, 2]
    flat = iy.astype(np.int64) + occ.shape[1] * iz.astype(np.int64)
    order = np.argsort(flat)
    flat_s = flat[order]
    uniq, start, counts = np.unique(flat_s, return_index=True, return_counts=True)
    half = max(1, int(round(0.5 * occ.voxel * ppm)))

    for u, s, c in zip(uniq, start, counts):
        sl = order[s : s + c]
        j = int(sl[np.argmax(occ.counts[sl])])
        cy, cz = float(occ.centers[j, 1]), float(occ.centers[j, 2])
        u_pix = int(round((cy - y0) * ppm))
        v_pix = int(round((z1 - cz) * ppm))  # z up
        col = colors_bgr[j]
        cv2.rectangle(
            img,
            (u_pix - half, v_pix - half),
            (u_pix + half, v_pix + half),
            (int(col[0]), int(col[1]), int(col[2])),
            -1,
        )

    for d in (-200, -100, 0, 100, 200, 300, 400):
        if y0 <= d <= y1:
            uu = int(round((d - y0) * ppm))
            cv2.line(img, (uu, 0), (uu, out_h - 1), (0, 180, 220), 1)

    cv2.putText(
        img,
        f"{title}  occupied={occ.centers.shape[0]}  voxel={occ.voxel:g}m",
        (12, out_h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
    )
    return img


def occ_semantic_colors(occ: OccupancyGrid, labels_to_bgr_fn) -> np.ndarray:
    return labels_to_bgr_fn(occ.labels)


def occ_height_colors(occ: OccupancyGrid) -> np.ndarray:
    return _height_to_bgr(occ.max_z, occ.z_range[0], occ.z_range[1])


def occ_binary_colors(occ: OccupancyGrid) -> np.ndarray:
    """Flat occupied color (amber)."""
    n = occ.centers.shape[0]
    cols = np.zeros((n, 3), dtype=np.uint8)
    cols[:] = (0, 200, 255)  # BGR amber
    return cols


def render_occ_camera_view(
    occ: OccupancyGrid,
    colors_bgr: np.ndarray,
    K: np.ndarray,
    dist5: np.ndarray,
    T_c_v: np.ndarray,
    width: int,
    height: int,
    *,
    title: str = "occ",
    max_voxels: int = 250000,
    seed: int = 0,
) -> np.ndarray:
    """Project occupied voxels into a camera-sized image (black bg, not overlaid)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    n = occ.centers.shape[0]
    if n == 0:
        cv2.putText(
            img,
            f"{title} empty",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 180, 180),
            2,
        )
        return img

    centers = occ.centers
    cols = colors_bgr
    if n > max_voxels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_voxels, replace=False)
        centers = centers[idx]
        cols = cols[idx]

    ones = np.ones((centers.shape[0], 1), dtype=np.float64)
    ph = np.hstack([centers.astype(np.float64), ones])
    pc = (T_c_v @ ph.T).T[:, :3]
    front = pc[:, 2] > 0.3
    pc = pc[front]
    cols = cols[front]
    if pc.shape[0] == 0:
        cv2.putText(
            img,
            f"{title} (no voxels in view)",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 180, 180),
            2,
        )
        return img

    uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist5)
    uv = uv.reshape(-1, 2)
    z = pc[:, 2]
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
        & np.isfinite(uv).all(axis=1)
    )
    uv = uv[inside]
    cols = cols[inside]
    z = z[inside]
    if uv.shape[0] == 0:
        cv2.putText(
            img,
            f"{title} (no voxels in view)",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 180, 180),
            2,
        )
        return img

    # Painter's algorithm: far → near
    order = np.argsort(-z)
    uv = uv[order]
    cols = cols[order]
    z = z[order]

    fx = float(K[0, 0])
    rad = np.clip((0.5 * occ.voxel * fx / np.maximum(z, 0.3)), 1.0, 10.0).astype(np.int32)

    # Fast path: write pixels; expand near voxels by their projected pixel radius
    # up to the per-voxel rad (clipped earlier to 1..10).  Must iterate up to
    # the *actual* max radius present, otherwise voxels with rad ∈ [4..10]
    # remain undersized.
    uu = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, width - 1)
    vv = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, height - 1)
    img[vv, uu] = cols
    max_r = int(rad.max()) if rad.size else 0
    for r in range(1, max_r + 1):
        sel = rad >= r
        if not np.any(sel):
            continue
        for du, dv in (
            (r, 0), (-r, 0), (0, r), (0, -r),
            (r, r), (r, -r), (-r, r), (-r, -r),
        ):
            u2 = np.clip(uu[sel] + du, 0, width - 1)
            v2 = np.clip(vv[sel] + dv, 0, height - 1)
            img[v2, u2] = cols[sel]

    cv2.putText(
        img,
        f"{title}  voxel={occ.voxel:g}m  n={uv.shape[0]}",
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 230, 230),
        2,
    )
    return img
