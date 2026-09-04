"""Voxelize point clouds into sparse occupancy grids (vehicle frame)."""

from dataclasses import dataclass

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

    # group boundaries (pre-filter so contiguous reduceat ranges are valid for
    # both counts AND majority-label computation).
    uniq, start_all, counts_all = np.unique(
        key_s, return_index=True, return_counts=True
    )

    # Majority label per voxel: per-class counts via np.add.reduceat on the
    # sorted label array, then argmax over class axis.  Class count loop is
    # small (Waymo ~ 30).  This replaces the previous "first-label" bug which
    # ignored all but the 1st point's label in each occupied voxel.
    if lab_s.size:
        n_class = int(lab_s.max()) + 1
    else:
        n_class = 1
    n_class = max(n_class, 1)
    per_class = np.empty((n_class, start_all.size), dtype=np.int32)
    for c in range(n_class):
        per_class[c] = np.add.reduceat(
            (lab_s == c).astype(np.int32, copy=False), start_all
        )
    maj_lab_all = per_class.argmax(axis=0).astype(np.int32)

    keep = counts_all >= int(min_points)
    start = start_all[keep]
    counts = counts_all[keep]
    if start.size == 0:
        return empty
    maj_lab = maj_lab_all[keep]
    max_z = np.maximum.reduceat(z_s, start).astype(np.float32)
    ijk = np.stack([ix_s[start], iy_s[start], iz_s[start]], axis=1).astype(np.int32)

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

