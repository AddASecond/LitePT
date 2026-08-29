#!/usr/bin/env python3
"""Build a radius-density-filtered OCC variant for an existing viewer scene."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OCC = load_module("density_compare_occ", "tools/robotruck_occupancy.py")
SAG = load_module("density_compare_static", "tools/robotruck_static_agg.py")
INFER = load_module("density_compare_infer", "tools/infer_robotruck_mongo_frame.py")
DENSITY = load_module("density_filter", "tools/robotruck_density_filter.py")


def asset(uri: str, dtype: str, shape) -> dict:
    return {"uri": uri, "dtype": dtype, "shape": list(shape), "byte_order": "little"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raw-cache", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--static-cache", required=True)
    ap.add_argument("--radius", type=float, default=0.6)
    ap.add_argument("--min-neighbors", type=int, default=4)
    ap.add_argument("--candidate-labels", type=int, nargs="+", default=[13, 14])
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    raw = Path(args.raw_cache).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    static_cache = Path(args.static_cache).resolve()
    index_path = scene / "index.json"
    index = json.loads(index_path.read_text())
    cached = np.load(static_cache)
    static = {
        "xyz_map": cached["xyz_map"].astype(np.float32),
        "labels": cached["labels"].astype(np.int32),
        "lidar_ids": cached["lidar_ids"].astype(np.int32),
    }
    keep, stats = DENSITY.radius_outlier_keep_mask(
        static["xyz_map"],
        static["labels"],
        radius=args.radius,
        min_neighbors=args.min_neighbors,
        candidate_labels=tuple(args.candidate_labels),
    )
    filtered = {key: value[keep] for key, value in static.items()}
    variant_id = "density_filtered"
    static_dir = scene / "static_agg"
    static_dir.mkdir(parents=True, exist_ok=True)
    filtered["xyz_map"].astype("<f4").tofile(static_dir / "density_xyz_map.f32.bin")
    filtered["labels"].astype("u1").tofile(static_dir / "density_labels.u8.bin")
    filtered["lidar_ids"].astype("u1").tofile(static_dir / "density_lidar_id.u8.bin")
    index.setdefault("static_agg_variants", {})[variant_id] = {
        "voxel": float(cached["voxel"][0]) if "voxel" in cached else 0.25,
        "n": int(len(filtered["xyz_map"])),
        "xyz_map": asset(
            "static_agg/density_xyz_map.f32.bin", "float32", filtered["xyz_map"].shape,
        ),
        "labels": asset(
            "static_agg/density_labels.u8.bin", "uint8", filtered["labels"].shape,
        ),
        "lidar_id": asset(
            "static_agg/density_lidar_id.u8.bin", "uint8", filtered["lidar_ids"].shape,
        ),
        "filter": stats,
    }
    total_before = 0
    total_after = 0

    for number, entry in enumerate(index["frames"], 1):
        ts = str(entry.get("timestamp") or entry["frame_id"])
        frame_dir = scene / "frames" / ts
        meta_path = frame_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        raw_meta = json.loads((raw / "frames" / ts / "frame.json").read_text())
        points = INFER.load_lidar_bin(
            raw / "frames" / ts / "lidar_merge.bin",
            num_cols=len(INFER.LIDAR_COLS),
        )
        xyz = points[:, :3].astype(np.float32)
        lidar = points[:, 6].astype(np.int32)
        labels = np.load(pred_dir / f"{ts}_pred.npy").astype(np.int32)
        pose = ((raw_meta.get("dependency") or {}).get("ego_pose") or {}).get("pose")
        if not pose:
            raise ValueError(f"{ts}: dependency.ego_pose.pose is missing")
        grid_meta = meta["grid"]
        xyz_s, lab_s, lid_s = SAG.static_in_vehicle(
            filtered,
            SAG.ego_pose_to_T_map_vehicle(pose),
            x_range=(grid_meta["x_range"][0] * 1.5, grid_meta["x_range"][1] * 1.5),
            y_range=tuple(grid_meta["y_range"]),
            z_range=(grid_meta["z_range"][0] - 2.0, grid_meta["z_range"][1] + 5.0),
        )
        merged_xyz, merged_lab, _, _ = SAG.merge_static_dynamic(
            xyz_s, lab_s, lid_s, xyz, labels, lidar,
        )
        grid = OCC.build_occupancy(
            merged_xyz,
            merged_lab,
            x_range=tuple(grid_meta["x_range"]),
            y_range=tuple(grid_meta["y_range"]),
            z_range=tuple(grid_meta["z_range"]),
            voxel=float(grid_meta["voxel"]),
            min_points=1,
        )
        prefix = f"frames/{ts}"
        arrays = {
            "ijk": ("occ_density_ijk.i32.bin", grid.ijk.astype("<i4"), "int32"),
            "labels": ("occ_density_labels.u8.bin", grid.labels.astype("u1"), "uint8"),
            "centers": ("occ_density_centers.f32.bin", grid.centers.astype("<f4"), "float32"),
            "counts": ("occ_density_counts.i32.bin", grid.counts.astype("<i4"), "int32"),
        }
        refs = {"n": int(len(grid.ijk))}
        for key, (name, array, dtype) in arrays.items():
            array.tofile(frame_dir / name)
            refs[key] = asset(f"{prefix}/{name}", dtype, array.shape)
        baseline = int((meta.get("assets", {}).get("occupancy") or {}).get("n", meta["stats"]["n_occ"]))
        meta.setdefault("assets", {}).setdefault("occupancy_variants", {})[variant_id] = refs
        meta.setdefault("stats", {})["density_filter"] = {
            "baseline_n_occ": baseline,
            "filtered_n_occ": refs["n"],
            "removed_occ": baseline - refs["n"],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        entry.setdefault("n_occ_variants", {})[variant_id] = refs["n"]
        total_before += baseline
        total_after += refs["n"]
        if number % 10 == 0 or number == len(index["frames"]):
            print(f"built {number}/{len(index['frames'])}", flush=True)

    variants = (index.get("occupancy_variants") or {}).get("variants") or []
    variants = [item for item in variants if item.get("id") != variant_id]
    if not any(item.get("id") == "litept" for item in variants):
        variants.insert(0, {"id": "litept", "name": "Before density filter"})
    variants.append({"id": variant_id, "name": "After density filter"})
    index["occupancy_variants"] = {"default": "litept", "variants": variants}
    stats.update({
        "frames": len(index["frames"]),
        "total_occ_before": total_before,
        "total_occ_after": total_after,
        "total_occ_removed": total_before-total_after,
    })
    index["density_filter_comparison"] = stats
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(json.dumps(stats, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
