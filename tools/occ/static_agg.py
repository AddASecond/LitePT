"""Clip-level static lidar aggregation (ego_pose → map frame, voxel cache)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Waymo-S LitePT indices (visualize.WAYMO_NAMES)
WAYMO_DYNAMIC_IDS = frozenset({0, 1, 2, 3, 4, 5, 6, 11, 12})
WAYMO_STATIC_IDS = frozenset(range(22)) - WAYMO_DYNAMIC_IDS
WAYMO_GROUND_IDS = frozenset({17, 18, 19, 20, 21})


def ground_aware_ego_keep_mask(xyz: np.ndarray, labels: np.ndarray, config: dict | None) -> tuple[np.ndarray, dict]:
    """Remove above-ground self returns only inside a configured XY footprint."""
    keep = np.ones(len(xyz), dtype=bool)
    if not config or not config.get("enabled", True) or not len(xyz):
        return keep, {"enabled": False, "removed": 0}
    x0, x1 = map(float, config["x_range"])
    y0, y1 = map(float, config["y_range"])
    margin = float(config.get("ground_fit_margin", 0.5))
    fit = (np.isin(np.asarray(labels).reshape(-1), list(WAYMO_GROUND_IDS))
           & (xyz[:, 0] >= x0 - 6.0) & (xyz[:, 0] <= x1 + 6.0)
           & (xyz[:, 1] >= y0 - 12.0) & (xyz[:, 1] <= y1 + 12.0)
           & ~((xyz[:, 0] >= x0 - margin) & (xyz[:, 0] <= x1 + margin)
               & (xyz[:, 1] >= y0 - margin) & (xyz[:, 1] <= y1 + margin)))
    fit_xyz = xyz[fit]
    if len(fit_xyz) >= 50:
        A = np.c_[fit_xyz[:, 0], fit_xyz[:, 1], np.ones(len(fit_xyz))]
        coef = np.linalg.lstsq(A, fit_xyz[:, 2], rcond=None)[0]
        for _ in range(3):
            residual = fit_xyz[:, 2] - A @ coef
            med = np.median(residual)
            mad = max(0.03, 1.4826 * np.median(np.abs(residual - med)))
            good = np.abs(residual - med) <= 2.5 * mad
            if good.sum() < 30:
                break
            coef = np.linalg.lstsq(A[good], fit_xyz[good, 2], rcond=None)[0]
    else:
        coef = np.array([0.0, 0.0, float(np.quantile(xyz[:, 2], 0.1))])
    ground_z = xyz[:, 0] * coef[0] + xyz[:, 1] * coef[1] + coef[2]
    height = xyz[:, 2] - ground_z
    inside_xy = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1)
                 & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
    remove = (inside_xy
              & (height > float(config.get("min_height", 0.35)))
              & (height < float(config.get("max_height", 4.0))))
    return ~remove, {"enabled": True, "removed": int(remove.sum()),
                     "ground_fit_points": int(fit.sum()),
                     "ground_plane": [float(v) for v in coef]}


def quat_to_rot(q: dict) -> np.ndarray:
    x, y, z, w = float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])
    n = max(1e-12, np.sqrt(x * x + y * y + z * z + w * w))
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def ego_pose_to_T_map_vehicle(pose: dict) -> np.ndarray:
    """4x4 map←vehicle from frame.json dependency.ego_pose.pose."""
    R = quat_to_rot(pose["orientation"])
    p = pose["position"]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return T


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    ph = np.hstack([xyz.astype(np.float64), ones])
    return (T @ ph.T).T[:, :3].astype(np.float32)


def points_in_oracle_boxes(
    xyz_veh: np.ndarray,
    objects: list[dict],
    *,
    margin: float = 0.3,
) -> np.ndarray:
    """Boolean mask: point inside any axis-aligned box in vehicle/imu frame.

    Oracle boxes give center_imu + size (l,w,h). We use AABB in vehicle frame
    expanded by margin (yaw ignored → slightly conservative exclusion).
    """
    n = xyz_veh.shape[0]
    if n == 0 or not objects:
        return np.zeros(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    for obj in objects:
        c = obj.get("center_imu")
        sz = obj.get("size")
        if c is None or sz is None:
            continue
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        # size: [length, width, height] along vehicle roughly
        l, w, h = float(sz[0]), float(sz[1]), float(sz[2])
        hx, hy, hz = 0.5 * l + margin, 0.5 * w + margin, 0.5 * h + margin
        inside = (
            (np.abs(xyz_veh[:, 0] - cx) <= hy)
            & (np.abs(xyz_veh[:, 1] - cy) <= hx)
            & (np.abs(xyz_veh[:, 2] - cz) <= hz)
        )
        mask |= inside
    return mask


def points_in_lidar_od_prelabel_boxes(
    xyz_veh: np.ndarray, objects: list[dict], *, margin: float = 0.15,
) -> np.ndarray:
    """Mask points in oriented lidar_od_prelabel boxes."""
    mask = np.zeros(len(xyz_veh), dtype=bool)
    for obj in objects:
        box = obj.get("box_lidar")
        if not isinstance(box, list) or len(box) < 7:
            continue
        x, y, z, length, width, height, yaw = map(float, box[:7])
        delta = xyz_veh - np.array([x, y, z], dtype=np.float32)
        c, s = np.cos(yaw), np.sin(yaw)
        local_x = c * delta[:, 0] + s * delta[:, 1]
        local_y = -s * delta[:, 0] + c * delta[:, 1]
        mask |= (
            (np.abs(local_x) <= 0.5 * length + margin)
            & (np.abs(local_y) <= 0.5 * width + margin)
            & (np.abs(delta[:, 2]) <= 0.5 * height + margin)
        )
    return mask


def static_mask_from_labels(
    labels: np.ndarray,
    xyz_veh: np.ndarray | None = None,
    oracle_objects: list[dict] | None = None,
    lidar_od_objects: list[dict] | None = None,
) -> np.ndarray:
    """True = keep for static aggregation."""
    lab = np.asarray(labels).astype(np.int64).reshape(-1)
    keep = np.isin(lab, list(WAYMO_STATIC_IDS))
    if xyz_veh is not None and oracle_objects:
        in_box = points_in_oracle_boxes(xyz_veh, oracle_objects)
        keep &= ~in_box
    if xyz_veh is not None and lidar_od_objects:
        keep &= ~points_in_lidar_od_prelabel_boxes(xyz_veh, lidar_od_objects)
    return keep


def voxel_downsample(
    xyz: np.ndarray,
    labels: np.ndarray,
    lidar_ids: np.ndarray | None = None,
    voxel: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Keep one point per voxel (first occurrence)."""
    if xyz.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        elab = np.zeros((0,), dtype=np.int32)
        eid = np.zeros((0,), dtype=np.int32) if lidar_ids is not None else None
        return empty, elab, eid
    v = max(1e-6, float(voxel))
    keys = np.floor(xyz.astype(np.float64) / v).astype(np.int64)
    # pack to 1d key (offset to positive)
    keys = keys - keys.min(axis=0, keepdims=True)
    # hash
    span = keys.max(axis=0) + 1
    flat = keys[:, 0] + span[0] * (keys[:, 1] + span[1] * keys[:, 2])
    _, uniq_idx = np.unique(flat, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    out_xyz = xyz[uniq_idx]
    out_lab = labels[uniq_idx].astype(np.int32)
    out_id = None if lidar_ids is None else lidar_ids[uniq_idx].astype(np.int32)
    return out_xyz, out_lab, out_id


def load_or_build_static_aggregate(
    clip_dir: Path,
    pred_dir: Path,
    timestamps: list[str],
    *,
    load_lidar_bin,
    lidar_cols: int,
    infer_frame=None,
    model=None,
    device=None,
    grid_size: float = 0.05,
    voxel: float = 0.2,
    cache_path: Path | None = None,
    use_oracle_boxes: bool = True,
    max_points_per_frame: int = 120000,
    seed: int = 0,
    ego_filter: dict | None = None,
    require_deskew: bool = True,
) -> dict:
    """Return dict with xyz_map, labels, lidar_ids (map frame), meta."""
    fingerprint_rows = []
    fingerprint_metas: dict[str, dict] = {}
    for ts in timestamps:
        frame = clip_dir / "frames" / ts
        meta_path = frame / "frame.json"
        pred_path = pred_dir / f"{ts}_pred.npy"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        fingerprint_metas[ts] = meta
        deskew = (((meta.get("dependency") or {}).get("sensors") or {}).get("lidar_merge_deskew") or {})
        fingerprint_rows.append({
            "timestamp": ts,
            "deskew_md5": deskew.get("md5"),
            "pose": ((meta.get("dependency") or {}).get("ego_pose") or {}).get("pose"),
            "od_source": (((meta.get("groundtruth") or {}).get("lidar_od_prelabel") or {}).get("source_version")),
            "pred_size": pred_path.stat().st_size if pred_path.is_file() else None,
            "pred_mtime_ns": pred_path.stat().st_mtime_ns if pred_path.is_file() else None,
        })
    expected_fingerprint = hashlib.sha256(json.dumps({
        "schema": "static_aggregate/v2", "frames": fingerprint_rows,
        "voxel": voxel, "grid_size": grid_size, "ego_filter": ego_filter or {},
        "use_oracle_boxes": use_oracle_boxes, "require_deskew": require_deskew,
    }, sort_keys=True).encode()).hexdigest()
    if cache_path is not None and cache_path.is_file():
        data = np.load(cache_path, allow_pickle=False)
        stored = str(data["source_fingerprint"][0]) if "source_fingerprint" in data else ""
        if stored == expected_fingerprint:
            return {
                "xyz_map": data["xyz_map"].astype(np.float32),
                "labels": data["labels"].astype(np.int32),
                "lidar_ids": data["lidar_ids"].astype(np.int32),
                "n_frames": int(data["n_frames"][0]) if "n_frames" in data else 0,
                "voxel": float(data["voxel"][0]) if "voxel" in data else voxel,
                "from_cache": True,
            }
        data.close()

    rng = np.random.default_rng(seed)
    acc_xyz: list[np.ndarray] = []
    acc_lab: list[np.ndarray] = []
    acc_lid: list[np.ndarray] = []
    used = 0

    for i, ts in enumerate(timestamps):
        fr = clip_dir / "frames" / ts
        lidar_path = fr / "lidar_merge.bin"
        if not lidar_path.is_file():
            continue
        meta = fingerprint_metas.get(ts)
        if meta is None:
            meta_path = fr / "frame.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
        deskew = (((meta.get("dependency") or {}).get("sensors") or {}).get("lidar_merge_deskew") or {})
        if require_deskew and not deskew.get("md5"):
            raise ValueError(f"{ts}: lidar_merge_deskew metadata is missing")
        pose = (meta.get("dependency") or {}).get("ego_pose", {}).get("pose")
        if not pose:
            continue
        T_map_v = ego_pose_to_T_map_vehicle(pose)

        pts = load_lidar_bin(lidar_path, num_cols=lidar_cols)
        coord = pts[:, :3].astype(np.float32)
        lidar_ids = pts[:, 6].astype(np.int32) if pts.shape[1] >= 7 else np.zeros(len(pts), np.int32)
        intensity = pts[:, 3]
        strength = np.tanh(intensity.reshape(-1, 1) / 255.0).astype(np.float32)

        pred_path = pred_dir / f"{ts}_pred.npy"
        if pred_path.is_file():
            pred = np.load(pred_path).astype(np.int64).reshape(-1)
            if pred.shape[0] != coord.shape[0]:
                if infer_frame is None:
                    continue
                pred = infer_frame(model, coord, strength, device, grid_size)
                np.save(pred_path, pred.astype(np.int32))
        else:
            if infer_frame is None:
                continue
            pred = infer_frame(model, coord, strength, device, grid_size)
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.save(pred_path, pred.astype(np.int32))

        oracle_objs = []
        if use_oracle_boxes:
            oracle_objs = (
                (meta.get("dependency") or {}).get("oracle", {}) or {}
            ).get("objects") or []

        lidar_od_objs = (
            ((meta.get("groundtruth") or {}).get("lidar_od_prelabel") or {})
        ).get("objects") or []

        ego_keep, _ = ground_aware_ego_keep_mask(coord, pred, ego_filter)
        keep = static_mask_from_labels(pred, coord, oracle_objs, lidar_od_objs) & ego_keep
        if not np.any(keep):
            continue

        xyz_s = coord[keep]
        lab_s = pred[keep].astype(np.int32)
        lid_s = lidar_ids[keep]
        if xyz_s.shape[0] > max_points_per_frame:
            idx = rng.choice(xyz_s.shape[0], size=max_points_per_frame, replace=False)
            xyz_s, lab_s, lid_s = xyz_s[idx], lab_s[idx], lid_s[idx]

        xyz_m = transform_points(xyz_s, T_map_v)
        acc_xyz.append(xyz_m)
        acc_lab.append(lab_s)
        acc_lid.append(lid_s)
        used += 1
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [static-agg] frame {i+1}/{len(timestamps)} kept_static={xyz_s.shape[0]}", flush=True)

    if not acc_xyz:
        out = {
            "xyz_map": np.zeros((0, 3), np.float32),
            "labels": np.zeros((0,), np.int32),
            "lidar_ids": np.zeros((0,), np.int32),
            "n_frames": 0,
            "voxel": voxel,
            "from_cache": False,
        }
    else:
        xyz = np.concatenate(acc_xyz, axis=0)
        lab = np.concatenate(acc_lab, axis=0)
        lid = np.concatenate(acc_lid, axis=0)
        print(f"  [static-agg] raw static points={xyz.shape[0]} from {used} frames; voxel={voxel}", flush=True)
        xyz, lab, lid = voxel_downsample(xyz, lab, lid, voxel=voxel)
        print(f"  [static-agg] after voxel={xyz.shape[0]}", flush=True)
        out = {
            "xyz_map": xyz,
            "labels": lab,
            "lidar_ids": lid,
            "n_frames": used,
            "voxel": voxel,
            "from_cache": False,
        }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            xyz_map=out["xyz_map"],
            labels=out["labels"],
            lidar_ids=out["lidar_ids"],
            n_frames=np.array([out["n_frames"]], dtype=np.int32),
            voxel=np.array([out["voxel"]], dtype=np.float32),
            ego_filter_json=np.array([json.dumps(ego_filter or {}, sort_keys=True)]),
            source_fingerprint=np.array([expected_fingerprint]),
        )
        print(f"  [static-agg] cached -> {cache_path}", flush=True)
    return out


def static_in_vehicle(
    agg: dict,
    T_map_vehicle: np.ndarray,
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform cached map static points into current vehicle frame + optional crop."""
    xyz_m = agg["xyz_map"]
    if xyz_m.shape[0] == 0:
        z = np.zeros((0, 3), np.float32)
        e = np.zeros((0,), np.int32)
        return z, e, e
    T_v_map = np.linalg.inv(T_map_vehicle)
    xyz_v = transform_points(xyz_m, T_v_map)
    lab = agg["labels"]
    lid = agg["lidar_ids"]
    m = np.ones(xyz_v.shape[0], dtype=bool)
    if x_range is not None:
        m &= (xyz_v[:, 0] >= x_range[0]) & (xyz_v[:, 0] <= x_range[1])
    if y_range is not None:
        m &= (xyz_v[:, 1] >= y_range[0]) & (xyz_v[:, 1] <= y_range[1])
    if z_range is not None:
        m &= (xyz_v[:, 2] >= z_range[0]) & (xyz_v[:, 2] <= z_range[1])
    return xyz_v[m], lab[m], lid[m]


def merge_static_dynamic(
    xyz_static: np.ndarray,
    lab_static: np.ndarray,
    lid_static: np.ndarray,
    xyz_frame: np.ndarray,
    lab_frame: np.ndarray,
    lid_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine clip-static + current-frame dynamic. Returns xyz, lab, lid, is_dynamic."""
    dyn = np.isin(lab_frame.astype(np.int64), list(WAYMO_DYNAMIC_IDS))
    # also keep non-static leftovers? only dynamic from frame
    xyz_d = xyz_frame[dyn]
    lab_d = lab_frame[dyn].astype(np.int32)
    lid_d = lid_frame[dyn].astype(np.int32)

    if xyz_static.shape[0] == 0 and xyz_d.shape[0] == 0:
        z = np.zeros((0, 3), np.float32)
        e = np.zeros((0,), np.int32)
        return z, e, e, np.zeros((0,), dtype=bool)

    parts_xyz = [p for p in (xyz_static, xyz_d) if p.shape[0]]
    parts_lab = [p for p in (lab_static, lab_d) if p.shape[0]]
    parts_lid = [p for p in (lid_static, lid_d) if p.shape[0]]
    flags = []
    if xyz_static.shape[0]:
        flags.append(np.zeros(xyz_static.shape[0], dtype=bool))
    if xyz_d.shape[0]:
        flags.append(np.ones(xyz_d.shape[0], dtype=bool))
    return (
        np.concatenate(parts_xyz, axis=0),
        np.concatenate(parts_lab, axis=0),
        np.concatenate(parts_lid, axis=0),
        np.concatenate(flags, axis=0),
    )
