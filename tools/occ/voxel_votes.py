"""Single-pass integer voxel voting; identical lowest-class tie breaking."""
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _majority(labels, starts, n_class):
    votes = np.zeros(n_class, dtype=np.int32)
    seen = np.full(n_class, -1, dtype=np.int64)
    result = np.zeros(len(starts), dtype=np.int32)
    for group in range(len(starts)):
        stop = starts[group + 1] if group + 1 < len(starts) else len(labels)
        best_count = 0
        best_label = 0
        for i in range(starts[group], stop):
            label = labels[i]
            # The NumPy reference only votes for nonnegative class IDs.
            if label < 0:
                continue
            if seen[label] != group:
                seen[label] = group
                votes[label] = 0
            votes[label] += 1
            count = votes[label]
            if count > best_count or (count == best_count and label < best_label):
                best_count = count
                best_label = label
        result[group] = best_label
    return result


majority = njit(cache=True, nogil=True)(_majority) if njit is not None else None


def _pack_voxels(xyz, labels, x0, x1, y0, y1, z0, z1, voxel, nx, ny, nz):
    keys = np.empty(len(xyz), dtype=np.int64)
    kept_labels = np.empty(len(xyz), dtype=np.int32)
    heights = np.empty(len(xyz), dtype=np.float32)
    count = 0
    for i in range(len(xyz)):
        x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        if not (x >= x0 and x < x1 and y >= y0 and y < y1 and z >= z0 and z < z1):
            continue
        # Explicit float32 steps match NumPy's float32-array/scalar arithmetic.
        ix = min(nx - 1, max(0, int(np.floor(np.float32(np.float32(x - x0) / voxel)))))
        iy = min(ny - 1, max(0, int(np.floor(np.float32(np.float32(y - y0) / voxel)))))
        iz = min(nz - 1, max(0, int(np.floor(np.float32(np.float32(z - z0) / voxel)))))
        keys[count] = ix + nx * (iy + ny * iz)
        kept_labels[count] = labels[i]
        heights[count] = z
        count += 1
    return keys[:count], kept_labels[:count], heights[:count]


pack_voxels = njit(cache=True, nogil=True)(_pack_voxels) if njit is not None else None
