"""Attach clip static_agg into an occ scene; rewrite per-frame points and/or occ.

Each frame:
  static_agg (map→vehicle via ego_pose) + frame dynamic
  → optional points export
  → optional occupancy voxelization (fixes scenes exported with empty static_agg)

Usage:
  export PYTHONPATH=./
  .venv_smoke/bin/python tools/repair_robotruck_scene_static.py \\
    --scene exp/robotruck/occ_scenes/rain_... --rewrite-occ --no-rewrite-points
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sag = _load("robotruck_static_agg", "tools/robotruck_static_agg.py")
_h = _load("infer_robotruck_mongo_frame", "tools/infer_robotruck_mongo_frame.py")
occmod = _load("robotruck_occupancy", "tools/robotruck_occupancy.py")


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.float32).tofile(path)


def write_u8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.uint8).tofile(path)


def write_i32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.int32).tofile(path)


def subsample_prefer_static(
    xyz_s: np.ndarray,
    lab_s: np.ndarray,
    lid_s: np.ndarray,
    xyz_d: np.ndarray,
    lab_d: np.ndarray,
    lid_d: np.ndarray,
    max_n: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep all dynamic; fill remaining budget with deterministic static stride."""
    n_d = int(xyz_d.shape[0])
    n_s = int(xyz_s.shape[0])
    if n_d + n_s <= max_n:
        parts_x = [p for p in (xyz_s, xyz_d) if p.shape[0]]
        parts_l = [p for p in (lab_s, lab_d) if p.shape[0]]
        parts_i = [p for p in (lid_s, lid_d) if p.shape[0]]
        if not parts_x:
            z = np.zeros((0, 3), np.float32)
            e = np.zeros((0,), np.uint8)
            return z, e, e
        return (
            np.concatenate(parts_x, axis=0),
            np.concatenate(parts_l, axis=0).astype(np.uint8),
            np.concatenate(parts_i, axis=0).astype(np.uint8),
        )

    budget_s = max(0, max_n - n_d)
    if budget_s <= 0:
        budget_s = max_n // 4
        rng = np.random.default_rng(seed)
        if n_d > max_n - budget_s:
            di = rng.choice(n_d, size=max_n - budget_s, replace=False)
            xyz_d, lab_d, lid_d = xyz_d[di], lab_d[di], lid_d[di]
            n_d = xyz_d.shape[0]
            budget_s = max_n - n_d

    if n_s > budget_s:
        step = int(np.ceil(n_s / budget_s))
        idx = np.arange(0, n_s, step)[:budget_s]
        xyz_s, lab_s, lid_s = xyz_s[idx], lab_s[idx], lid_s[idx]

    parts_x = [p for p in (xyz_s, xyz_d) if p.shape[0]]
    parts_l = [p for p in (lab_s, lab_d) if p.shape[0]]
    parts_i = [p for p in (lid_s, lid_d) if p.shape[0]]
    return (
        np.concatenate(parts_x, axis=0),
        np.concatenate(parts_l, axis=0).astype(np.uint8),
        np.concatenate(parts_i, axis=0).astype(np.uint8),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Path to occ scene root")
    ap.add_argument(
        "--static-npz",
        default="",
        help="Override static_agg npz (default: clip_video/.../static_voxel0.25_s5.npz)",
    )
    ap.add_argument("--backup-root", default="data/robotruck_clips_backup")
    ap.add_argument("--max-export-points", type=int, default=200000)
    ap.add_argument("--occ-voxel", type=float, default=0.0, help="0 = use meta.grid.voxel")
    ap.add_argument("--occ-min-points", type=int, default=1)
    ap.add_argument(
        "--rewrite-points",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--rewrite-occ",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild occ from static_agg⊕dyn (needed if export used empty agg)",
    )
    args = ap.parse_args()
    if not args.rewrite_points and not args.rewrite_occ:
        raise SystemExit("Nothing to do: enable --rewrite-points and/or --rewrite-occ")

    scene = Path(args.scene).resolve()
    idx_path = scene / "index.json"
    index = json.loads(idx_path.read_text())
    clip_id = index.get("clip_id") or index.get("clip") or scene.name

    npz_path = (
        Path(args.static_npz).resolve()
        if args.static_npz
        else (
            ROOT
            / "exp/robotruck/clip_video"
            / clip_id
            / "static_agg"
            / "static_voxel0.25_s5.npz"
        )
    )
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    xyz_map = data["xyz_map"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    lidar_ids = data["lidar_ids"].astype(np.int32)
    voxel = float(data["voxel"][0]) if "voxel" in data else 0.25
    agg = {"xyz_map": xyz_map, "labels": labels, "lidar_ids": lidar_ids, "voxel": voxel}
    n = int(xyz_map.shape[0])
    if n == 0:
        raise RuntimeError(
            f"static_agg is empty: {npz_path}. Rebuild with export/aggregate before repair."
        )
    out_dir = scene / "static_agg"
    write_f32(out_dir / "xyz_map.f32.bin", xyz_map)
    write_u8(out_dir / "labels.u8.bin", labels.astype(np.uint8))
    write_u8(out_dir / "lidar_id.u8.bin", lidar_ids.astype(np.uint8))
    index["static_agg"] = {
        "voxel": voxel,
        "n": n,
        "source_npz": str(npz_path.relative_to(ROOT))
        if str(npz_path).startswith(str(ROOT))
        else str(npz_path),
        "xyz_map": {
            "uri": "static_agg/xyz_map.f32.bin",
            "dtype": "float32",
            "shape": [n, 3],
        },
        "labels": {"uri": "static_agg/labels.u8.bin", "dtype": "uint8", "shape": [n]},
        "lidar_id": {
            "uri": "static_agg/lidar_id.u8.bin",
            "dtype": "uint8",
            "shape": [n],
        },
        "note": "Clip-level static in map frame; each frame only applies ego_pose.",
    }
    idx_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote static_agg bins n={n} voxel={voxel} → {out_dir}")

    clip_dir = (ROOT / args.backup_root / clip_id).resolve()
    pred_dir = (ROOT / "exp/robotruck/clip_video" / clip_id / "preds").resolve()
    frames = index.get("frames") or []
    for i, fr in enumerate(frames):
        meta_path = scene / fr["meta_uri"]
        meta = json.loads(meta_path.read_text())
        ts = str(meta.get("timestamp") or meta.get("frame_id"))
        pose = meta.get("ego_pose")
        if not pose:
            print(f"  skip {ts}: no ego_pose")
            continue
        T_map_v = sag.ego_pose_to_T_map_vehicle(pose)
        grid_meta = meta.get("grid") or {}
        x_range = tuple(grid_meta.get("x_range") or meta["x_range"])
        y_range = tuple(grid_meta.get("y_range") or meta["y_range"])
        z_range = tuple(grid_meta.get("z_range") or meta["z_range"])
        occ_voxel = (
            float(args.occ_voxel)
            if args.occ_voxel > 0
            else float(grid_meta.get("voxel") or meta.get("voxel") or 0.2)
        )
        xyz_s, lab_s, lid_s = sag.static_in_vehicle(
            agg,
            T_map_v,
            x_range=(x_range[0] * 1.5, x_range[1] * 1.5),
            y_range=y_range,
            z_range=(z_range[0] - 2.0, z_range[1] + 5.0),
        )

        fr_dir = clip_dir / "frames" / ts
        lidar_path = fr_dir / "lidar_merge.bin"
        pred_path = pred_dir / f"{ts}_pred.npy"
        if not lidar_path.is_file() or not pred_path.is_file():
            print(f"  skip {ts}: missing lidar/pred")
            continue
        pts = _h.load_lidar_bin(lidar_path, num_cols=len(_h.LIDAR_COLS))
        coord = pts[:, :3].astype(np.float32)
        lidar_ids_f = pts[:, 6].astype(np.int32)
        pred = np.load(pred_path).astype(np.int64).reshape(-1)
        if pred.shape[0] != coord.shape[0]:
            print(f"  skip {ts}: pred size mismatch")
            continue
        vis_xyz, vis_lab, vis_lid, _ = sag.merge_static_dynamic(
            xyz_s, lab_s, lid_s, coord, pred, lidar_ids_f
        )
        dyn = np.isin(pred, list(sag.WAYMO_DYNAMIC_IDS))
        xyz_d = coord[dyn]
        lab_d = pred[dyn].astype(np.int32)
        lid_d = lidar_ids_f[dyn]

        frame_dir = meta_path.parent
        n_pts = int((meta.get("stats") or {}).get("n_points_exported") or 0)
        n_occ = int((meta.get("stats") or {}).get("n_occ") or meta.get("n_occ") or 0)

        if args.rewrite_points:
            p_xyz, p_lab, p_lid = subsample_prefer_static(
                xyz_s, lab_s, lid_s, xyz_d, lab_d, lid_d, args.max_export_points
            )
            write_f32(frame_dir / "points_xyz.f32.bin", p_xyz)
            write_u8(frame_dir / "points_labels.u8.bin", p_lab)
            write_u8(frame_dir / "points_lidar_id.u8.bin", p_lid)
            n_pts = int(p_xyz.shape[0])
            if meta.get("assets") and meta["assets"].get("points"):
                meta["assets"]["points"]["n"] = n_pts
                for k, shape in (
                    ("xyz", [n_pts, 3]),
                    ("labels", [n_pts]),
                    ("lidar_id", [n_pts]),
                ):
                    if k in meta["assets"]["points"]:
                        meta["assets"]["points"][k]["shape"] = shape
            if meta.get("points"):
                meta["points"]["n"] = n_pts

        if args.rewrite_occ:
            occ = occmod.build_occupancy(
                vis_xyz,
                vis_lab,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                voxel=occ_voxel,
                min_points=args.occ_min_points,
            )
            write_f32(frame_dir / "occ_centers.f32.bin", occ.centers)
            write_u8(frame_dir / "occ_labels.u8.bin", occ.labels.astype(np.uint8))
            write_i32(frame_dir / "occ_ijk.i32.bin", occ.ijk)
            write_i32(frame_dir / "occ_counts.i32.bin", occ.counts)
            n_occ = int(occ.centers.shape[0])
            if meta.get("grid"):
                meta["grid"]["voxel"] = float(occ.voxel)
                meta["grid"]["shape"] = list(occ.shape)
            meta["voxel"] = float(occ.voxel)
            meta["occ_shape"] = list(occ.shape)
            meta["n_occ"] = n_occ
            if meta.get("assets") and meta["assets"].get("occupancy"):
                meta["assets"]["occupancy"]["n"] = n_occ
                for k, shape in (
                    ("ijk", [n_occ, 3]),
                    ("labels", [n_occ]),
                    ("centers", [n_occ, 3]),
                    ("counts", [n_occ]),
                ):
                    if k in meta["assets"]["occupancy"]:
                        meta["assets"]["occupancy"][k]["shape"] = shape

        if meta.get("stats") is None:
            meta["stats"] = {}
        meta["stats"]["n_static_roi"] = int(xyz_s.shape[0])
        meta["stats"]["n_vis_points"] = int(vis_xyz.shape[0])
        if args.rewrite_points:
            meta["stats"]["n_points_exported"] = n_pts
            meta["stats"]["points_source"] = "static_agg+dynamic"
        if args.rewrite_occ:
            meta["stats"]["n_occ"] = n_occ
            meta["stats"]["occ_source"] = "static_agg+dynamic"
        if "n_occ" in fr:
            fr["n_occ"] = n_occ
        if "n_points" in fr and args.rewrite_points:
            fr["n_points"] = n_pts

        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        if (i + 1) % 20 == 0 or i == 0 or i + 1 == len(frames):
            print(
                f"  [{i+1}/{len(frames)}] {ts} static={xyz_s.shape[0]} dyn={int(dyn.sum())} "
                f"vis={vis_xyz.shape[0]} occ={n_occ} pts={n_pts}",
                flush=True,
            )

    idx_path.write_text(json.dumps(index, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
