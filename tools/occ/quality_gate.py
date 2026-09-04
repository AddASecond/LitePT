"""Prod: clip-level geometry quality gate (single SoT for export + offline scan).

Rejects corroborated multi-lidar height layering or high-confidence pose-alignment failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import static_agg as sag


def _load(clip_dir: Path, timestamp: str) -> tuple[np.ndarray, dict]:
    frame = clip_dir / "frames" / timestamp
    meta = json.loads((frame / "frame.json").read_text())
    points = np.fromfile(frame / "lidar_merge.bin", np.float32).reshape(-1, 7)
    deskew = (((meta.get("dependency") or {}).get("sensors") or {}).get("lidar_merge_deskew") or {})
    if not deskew.get("md5"):
        raise ValueError(f"{timestamp}: lidar_merge_deskew metadata is missing")
    return points, meta

def _lowest_cells(points: np.ndarray, lidar_id: int) -> dict[tuple[int, int], float]:
    keep = (
        (points[:, 6].astype(np.int32) == lidar_id)
        & (points[:, 0] > -20) & (points[:, 0] < 20)
        & (points[:, 1] > 5) & (points[:, 1] < 100)
        & (points[:, 2] > -5) & (points[:, 2] < 1.5)
    )
    selected = points[keep]
    if not len(selected):
        return {}
    cells = np.floor(selected[:, :2]).astype(np.int32)
    order = np.lexsort((selected[:, 2], cells[:, 1], cells[:, 0]))
    unique, first = np.unique(cells[order], axis=0, return_index=True)
    return {tuple(cell): float(value) for cell, value in zip(unique, selected[order, 2][first])}

def _layer_score(points: np.ndarray, min_overlap: int) -> dict:
    cells = {lid: _lowest_cells(points, lid) for lid in (1, 2, 14)}
    pairwise = {}
    for left, right in ((1, 2), (1, 14), (2, 14)):
        overlap = cells[left].keys() & cells[right].keys()
        if len(overlap) < min_overlap:
            continue
        delta = np.array([cells[left][key] - cells[right][key] for key in overlap])
        pairwise[f"{left}-{right}"] = {
            "overlap": len(overlap),
            "median_dz": float(np.median(delta)),
        }
    reference = pairwise.get("1-2")
    side = [pairwise.get("1-14"), pairwise.get("2-14")]
    valid_side = [abs(item["median_dz"]) for item in side if item]
    return {
        "score": max(valid_side) if len(valid_side) == 2 and reference else None,
        "reference_1_2": abs(reference["median_dz"]) if reference else None,
        "pairwise": pairwise,
    }

def _structural_map(points: np.ndarray, meta: dict, voxel: float, max_points: int) -> np.ndarray:
    xyz = points[:, :3]
    keep = (
        (xyz[:, 0] > -25) & (xyz[:, 0] < 25)
        & (xyz[:, 1] > -20) & (xyz[:, 1] < 120)
        & (xyz[:, 2] > -1.1) & (xyz[:, 2] < 4)
    )
    objects = (((meta.get("groundtruth") or {}).get("lidar_od_prelabel") or {}).get("objects") or [])
    keep &= ~sag.points_in_lidar_od_prelabel_boxes(xyz, objects)
    xyz = xyz[keep]
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, first = np.unique(keys, axis=0, return_index=True)
    xyz = xyz[np.sort(first)]
    if len(xyz) > max_points:
        xyz = xyz[np.linspace(0, len(xyz)-1, max_points, dtype=np.int64)]
    pose = ((meta.get("dependency") or {}).get("ego_pose") or {}).get("pose")
    if not pose:
        raise ValueError("dependency.ego_pose.pose is missing")
    return sag.transform_points(xyz, sag.ego_pose_to_T_map_vehicle(pose)).astype(np.float64)

def _trimmed_loss(tree: cKDTree, points: np.ndarray, max_distance: float) -> float:
    distance = tree.query(points, workers=-1)[0]
    distance = distance[distance < max_distance]
    if len(distance) < 100:
        return float("inf")
    count = max(100, int(0.7 * len(distance)))
    return float(np.median(np.partition(distance, count-1)[:count]))

def assess_clip_geometry(
    clip_dir: Path,
    timestamps: list[str],
    *,
    sample_frames: int = 5,
    pose_pair_gap_seconds: float = 0.5,
    layer_threshold: float = 0.15,
    layer_reference_threshold: float = 0.05,
    layer_min_overlap: int = 300,
    pose_shift_threshold: float = 0.40,
    pose_min_improved_pairs: int = 4,
    pose_min_sign_consistency: float = 0.80,
    pose_min_relative_improvement: float = 0.05,
    pose_max_best_loss: float = 0.18,
    pose_max_layer_score: float = 0.01,
) -> dict:
    """Reject only strongly corroborated layering or pose-alignment failures."""
    if len(timestamps) < 3:
        return {"allow_occ": False, "reasons": ["insufficient_frames"], "warnings": []}
    frame_cache: dict[str, tuple[np.ndarray, dict]] = {}

    def load_cached(timestamp: str) -> tuple[np.ndarray, dict]:
        hit = frame_cache.get(timestamp)
        if hit is None:
            hit = _load(clip_dir, timestamp)
            frame_cache[timestamp] = hit
        return hit

    sample_index = np.linspace(0, len(timestamps)-1, min(sample_frames, len(timestamps)), dtype=int)
    layer_rows = []
    for index in sample_index:
        points, _ = load_cached(timestamps[index])
        layer_rows.append(_layer_score(points, layer_min_overlap))
    valid_layer = [row for row in layer_rows if row["score"] is not None]
    layer_score = float(np.median([row["score"] for row in valid_layer])) if valid_layer else None
    reference_score = float(np.median([row["reference_1_2"] for row in valid_layer])) if valid_layer else None
    layer_signal = (
        len(valid_layer) >= 3
        and layer_score is not None and layer_score > layer_threshold
        and reference_score is not None and reference_score < layer_reference_threshold
    )

    timestamp_ns = np.asarray([int(ts) for ts in timestamps], dtype=np.int64)
    target_gap_ns = int(pose_pair_gap_seconds * 1e9)
    candidates = []
    for index, timestamp in enumerate(timestamp_ns):
        next_index = int(np.searchsorted(timestamp_ns, timestamp + target_gap_ns))
        if next_index >= len(timestamp_ns):
            continue
        candidates.append((index, next_index))
    if not candidates:
        return {"allow_occ": False, "reasons": ["insufficient_pose_time_span"], "warnings": []}
    selected = np.linspace(0, len(candidates)-1, min(sample_frames, len(candidates)), dtype=int)
    pose_pairs = [candidates[index] for index in selected]
    pose_rows = []
    shifts = np.linspace(-1.2, 1.2, 25)
    for index, next_index in pose_pairs:
        first_points, first_meta = load_cached(timestamps[index])
        next_points, next_meta = load_cached(timestamps[next_index])
        first_map = _structural_map(first_points, first_meta, 0.4, 8000)
        next_map = _structural_map(next_points, next_meta, 0.4, 8000)
        p0 = sag.ego_pose_to_T_map_vehicle(first_meta["dependency"]["ego_pose"]["pose"])[:3, 3]
        p1 = sag.ego_pose_to_T_map_vehicle(next_meta["dependency"]["ego_pose"]["pose"])[:3, 3]
        direction = p1 - p0
        direction /= max(1e-9, np.linalg.norm(direction))
        tree = cKDTree(first_map)
        losses = [_trimmed_loss(tree, next_map + shift*direction, 1.2) for shift in shifts]
        best = int(np.argmin(losses))
        loss_at_pose = float(losses[len(shifts)//2])
        best_loss = float(losses[best])
        relative_improvement = max(0.0, (loss_at_pose-best_loss) / max(1e-9, loss_at_pose))
        pose_rows.append({
            "delta_seconds": (timestamp_ns[next_index] - timestamp_ns[index]) / 1e9,
            "shift": float(shifts[best]),
            "loss_at_pose": loss_at_pose,
            "best_loss": best_loss,
            "relative_improvement": relative_improvement,
            "improved": bool(relative_improvement > 0.02),
        })
    median_shift = float(np.median(np.abs([row["shift"] for row in pose_rows])))
    improved_pairs = sum(row["improved"] for row in pose_rows)
    significant_shifts = [row["shift"] for row in pose_rows if abs(row["shift"]) >= 0.2]
    sign_consistency = (
        max(sum(shift > 0 for shift in significant_shifts), sum(shift < 0 for shift in significant_shifts))
        / max(1, len(significant_shifts))
    )
    median_relative_improvement = float(np.median([row["relative_improvement"] for row in pose_rows]))
    median_best_loss = float(np.median([row["best_loss"] for row in pose_rows]))
    pose_signal = median_shift > pose_shift_threshold and improved_pairs >= pose_min_improved_pairs
    high_confidence_pose_bad = (
        pose_signal
        and sign_consistency >= pose_min_sign_consistency
        and median_relative_improvement >= pose_min_relative_improvement
        and median_best_loss <= pose_max_best_loss
        and layer_score is not None
        and layer_score <= pose_max_layer_score
    )
    corroborated_layer_bad = layer_signal and pose_signal

    reasons = []
    if corroborated_layer_bad:
        reasons.append(f"layering_with_pose_inconsistency:{layer_score:.3f}m/{median_shift:.3f}m")
    if high_confidence_pose_bad:
        reasons.append(f"high_confidence_pose_drift:{median_shift:.3f}m/{improved_pairs}pairs")
    warnings = []
    if layer_signal and not corroborated_layer_bad:
        warnings.append(f"uncorroborated_lidar_height_offset:{layer_score:.3f}m")
    if pose_signal and not high_confidence_pose_bad and not corroborated_layer_bad:
        warnings.append(f"low_confidence_pose_alignment_signal:{median_shift:.3f}m")
    return {
        "allow_occ": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "layer": {
            "median_score": layer_score,
            "median_reference_1_2": reference_score,
            "valid_frames": len(valid_layer),
            "rows": layer_rows,
        },
        "pose": {
            "median_abs_shift": median_shift,
            "improved_pairs": improved_pairs,
            "sign_consistency": sign_consistency,
            "median_relative_improvement": median_relative_improvement,
            "median_best_loss": median_best_loss,
            "rows": pose_rows,
        },
        "thresholds": {
            "layer": layer_threshold,
            "layer_reference_1_2": layer_reference_threshold,
            "pose_pair_gap_seconds": pose_pair_gap_seconds,
            "pose_shift": pose_shift_threshold,
            "pose_min_improved_pairs": pose_min_improved_pairs,
            "pose_min_sign_consistency": pose_min_sign_consistency,
            "pose_min_relative_improvement": pose_min_relative_improvement,
            "pose_max_best_loss": pose_max_best_loss,
            "pose_max_layer_score": pose_max_layer_score,
        },
    }
