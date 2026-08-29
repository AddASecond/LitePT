"""Traditional density filters for aggregated Robotruck point clouds."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def radius_outlier_keep_mask(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    radius: float = 0.6,
    min_neighbors: int = 4,
    candidate_labels: tuple[int, ...] = (13, 14),
) -> tuple[np.ndarray, dict]:
    """Return a keep mask using fixed-radius neighbor density.

    min_neighbors includes the query point itself. Only candidate semantic
    classes are filtered, preserving ground, signs, poles, cones and curbs.
    """
    points = np.asarray(xyz, dtype=np.float32)
    semantic = np.asarray(labels).reshape(-1).astype(np.int32)
    if len(points) != len(semantic):
        raise ValueError("xyz and labels length mismatch")
    if radius <= 0 or min_neighbors < 2:
        raise ValueError("radius must be > 0 and min_neighbors must be >= 2")

    keep = np.ones(len(points), dtype=bool)
    candidate = np.isin(semantic, candidate_labels)
    candidate_index = np.flatnonzero(candidate)
    if len(candidate_index):
        tree = cKDTree(points)
        distance = tree.query(
            points[candidate_index],
            k=min_neighbors,
            distance_upper_bound=radius,
            workers=-1,
        )[0]
        kth = distance[:, -1]
        keep[candidate_index] = np.isfinite(kth)

    removed_by_label = {
        str(label): int(np.sum((~keep) & (semantic == label)))
        for label in candidate_labels
    }
    stats = {
        "algorithm": "radius_outlier_removal/v1",
        "radius_m": float(radius),
        "min_neighbors_including_self": int(min_neighbors),
        "candidate_labels": list(candidate_labels),
        "input_points": int(len(points)),
        "candidate_points": int(len(candidate_index)),
        "removed_points": int(np.sum(~keep)),
        "kept_points": int(np.sum(keep)),
        "removed_by_label": removed_by_label,
    }
    return keep, stats
