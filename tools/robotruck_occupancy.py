"""Voxelize point clouds into occupancy grids for Robotruck visualization.

Occupied voxels are derived from (clip-static + frame-dynamic) points in the
vehicle frame. Each occupied cell stores majority semantic label, point count,
and max height — suitable for BEV / side occupancy panels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OccupancyGrid:
    """Sparse occupied voxels in vehicle frame."""

    # voxel index (i_lat, i_fwd, i_z) and centers
    ijk: np.ndarray  # Nx3 int32
    centers: np.ndarray  # Nx3 float32 (x,y,z) vehicle
    labels: np.ndarray  # N int32 majority class
    counts: np.ndarray  # N int32
    max_z: np.ndarray  # N float32
    voxel: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    shape: tuple[int, int, int]  # (nx, ny, nz)


def _range_to_size(lo: float, hi: float, voxel: float) -> int:
    return max(1, int(np.ceil((hi - lo) / max(1e-6, voxel))))


def build_occupancy(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    voxel: float = 0.4,
    min_points: int = 1,
) -> OccupancyGrid:
    """Voxelize points; keep cells with >= min_points as occupied."""
    v = max(1e-6, float(voxel))
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    nx = _range_to_size(x0, x1, v)
    ny = _range_to_size(y0, y1, v)
    nz = _range_to_size(z0, z1, v)

    empty = OccupancyGrid(
        ijk=np.zeros((0, 3), np.int32),
        centers=np.zeros((0, 3), np.float32),
        labels=np.zeros((0,), np.int32),
        counts=np.zeros((0,), np.int32),
        max_z=np.zeros((0,), np.float32),
        voxel=v,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        shape=(nx, ny, nz),
    )
    if xyz is None or xyz.shape[0] == 0:
        return empty

    xyz = np.asarray(xyz, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    m = (
        (xyz[:, 0] >= x0)
        & (xyz[:, 0] < x1)
        & (xyz[:, 1] >= y0)
        & (xyz[:, 1] < y1)
        & (xyz[:, 2] >= z0)
        & (xyz[:, 2] < z1)
    )
    xyz = xyz[m]
    lab = lab[m]
    if xyz.shape[0] == 0:
        return empty

    ix = np.floor((xyz[:, 0] - x0) / v).astype(np.int32)
    iy = np.floor((xyz[:, 1] - y0) / v).astype(np.int32)
    iz = np.floor((xyz[:, 2] - z0) / v).astype(np.int32)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)

    # pack key
    key = ix.astype(np.int64) + nx * (iy.astype(np.int64) + ny * iz.astype(np.int64))
    order = np.argsort(key)
    key_s = key[order]
    lab_s = lab[order]
    z_s = xyz[order, 2]
    ix_s, iy_s, iz_s = ix[order], iy[order], iz[order]

    # group boundaries
    uniq, start, counts = np.unique(key_s, return_index=True, return_counts=True)
    keep = counts >= int(min_points)
    uniq = uniq[keep]
    start = start[keep]
    counts = counts[keep]
    if uniq.size == 0:
        return empty

    maj_lab = np.empty(uniq.size, dtype=np.int32)
    max_z = np.empty(uniq.size, dtype=np.float32)
    ijk = np.empty((uniq.size, 3), dtype=np.int32)
    for i, (s, c) in enumerate(zip(start, counts)):
        sl = slice(s, s + c)
        vals, cnts = np.unique(lab_s[sl], return_counts=True)
        maj_lab[i] = int(vals[np.argmax(cnts)])
        max_z[i] = float(z_s[sl].max())
        ijk[i, 0] = int(ix_s[s])
        ijk[i, 1] = int(iy_s[s])
        ijk[i, 2] = int(iz_s[s])

    centers = np.stack(
        [
            x0 + (ijk[:, 0].astype(np.float32) + 0.5) * v,
            y0 + (ijk[:, 1].astype(np.float32) + 0.5) * v,
            z0 + (ijk[:, 2].astype(np.float32) + 0.5) * v,
        ],
        axis=1,
    ).astype(np.float32)

    return OccupancyGrid(
        ijk=ijk,
        centers=centers,
        labels=maj_lab,
        counts=counts.astype(np.int32),
        max_z=max_z,
        voxel=v,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        shape=(nx, ny, nz),
    )


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

    # Collapse Z: one cell per (ix,iy) — keep max-count (or any)
    ix = occ.ijk[:, 0]
    iy = occ.ijk[:, 1]
    flat = ix.astype(np.int64) + occ.shape[0] * iy.astype(np.int64)
    order = np.argsort(flat)
    flat_s = flat[order]
    uniq, start, counts = np.unique(flat_s, return_index=True, return_counts=True)

    half = max(1, int(round(0.5 * occ.voxel * ppm)))
    for u, s, c in zip(uniq, start, counts):
        sl = order[s : s + c]
        # pick voxel with most points among z-stack
        j = int(sl[np.argmax(occ.counts[sl])])
        cx, cy = float(occ.centers[j, 0]), float(occ.centers[j, 1])
        u_pix = int(round((cy - y0) * ppm))
        v_pix = int(round((cx - x0) * ppm))
        col = colors_bgr[j]
        cv2.rectangle(
            img,
            (u_pix - half, v_pix - half),
            (u_pix + half, v_pix + half),
            (int(col[0]), int(col[1]), int(col[2])),
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
