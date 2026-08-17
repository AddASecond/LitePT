"""Export Robotruck frame packages for the occ frontend viewer.

Separates data production from visualization:
  - clean camera JPGs (no point overlay)
  - 0.1m occupancy voxels (centers + labels)
  - optional point cloud for toggle in the viewer
  - camera intrinsics/extrinsics + ego_pose

Usage:
  export PYTHONPATH=./
  .venv_smoke/bin/python tools/export_robotruck_occ_scene.py \\
    --clip stop_1784423032302844849_vehicle-V002-20260719_090818 \\
    --stride 2 --max-frames 3 --reuse-pred --occ-voxel 0.1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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


_h = _load("infer_robotruck_mongo_frame", "tools/infer_robotruck_mongo_frame.py")
sag = _load("robotruck_static_agg", "tools/robotruck_static_agg.py")
occmod = _load("robotruck_occupancy", "tools/robotruck_occupancy.py")
vis = _load("visualize_mod", "visualize.py")

CAM_ORDER = [
    "camera1",
    "camera2",
    "camera3",
    "camera4",
    "camera5",
    "camera6",
    "camera7",
    "camera8",
    "camera9",
    "camera17",
]


def list_clip_frames(clip_dir: Path) -> list[str]:
    idx_path = clip_dir / "frames_index.json"
    if idx_path.is_file():
        entries = json.loads(idx_path.read_text())
        return [str(e["timestamp"]) for e in entries if e.get("has_lidar")]
    return [
        p.name
        for p in sorted((clip_dir / "frames").iterdir())
        if (p / "lidar_merge.bin").is_file()
    ]


def parse_camera(cam_doc: dict):
    K = np.asarray(cam_doc["intrinsic"]["intrinsic"], dtype=np.float64)
    dist = np.asarray(cam_doc["intrinsic"]["distortion"], dtype=np.float64).reshape(-1)
    dist5 = np.zeros(5, dtype=np.float64)
    dist5[: min(5, dist.size)] = dist[:5]
    T_v_c = np.asarray(cam_doc["extrinsic"]["transformation"], dtype=np.float64)
    T_c_v = np.linalg.inv(T_v_c)
    w = int(cam_doc["intrinsic"]["width"])
    h = int(cam_doc["intrinsic"]["height"])
    return K, dist5, T_c_v, T_v_c, w, h


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.asarray(arr, dtype=np.float32).reshape(-1).tobytes())


def write_u8(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.asarray(arr, dtype=np.uint8).reshape(-1).tobytes())


def write_i32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.asarray(arr, dtype=np.int32).reshape(-1).tobytes())


def export_frame(
    *,
    clip_dir: Path,
    out_frame: Path,
    ts: str,
    pred_dir: Path,
    static_agg: dict | None,
    model,
    device,
    grid_size: float,
    reuse_pred: bool,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    occ_voxel: float,
    occ_min_points: int,
    export_points: bool,
    max_export_points: int,
) -> dict:
    fr = clip_dir / "frames" / ts
    meta = json.loads((fr / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]

    pts = _h.load_lidar_bin(fr / "lidar_merge.bin", num_cols=len(_h.LIDAR_COLS))
    coord = pts[:, :3].astype(np.float32)
    lidar_ids = pts[:, 6].astype(np.int32)
    intensity = pts[:, 3]
    strength = np.tanh(intensity.reshape(-1, 1) / 255.0).astype(np.float32)

    pred_path = pred_dir / f"{ts}_pred.npy"
    if reuse_pred and pred_path.is_file():
        pred = np.load(pred_path).astype(np.int64).reshape(-1)
        if pred.shape[0] != coord.shape[0]:
            pred = _h.infer_frame(model, coord, strength, device, grid_size)
            np.save(pred_path, pred.astype(np.int32))
    else:
        pred = _h.infer_frame(model, coord, strength, device, grid_size)
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, pred.astype(np.int32))

    lab_s = np.zeros((0,), np.int32)
    xyz_s = np.zeros((0, 3), np.float32)
    if static_agg is not None and static_agg["xyz_map"].shape[0] > 0:
        pose = (meta.get("dependency") or {}).get("ego_pose", {}).get("pose")
        if pose:
            T_map_v = sag.ego_pose_to_T_map_vehicle(pose)
            xyz_s, lab_s, lid_s = sag.static_in_vehicle(
                static_agg,
                T_map_v,
                x_range=(x_range[0] * 1.5, x_range[1] * 1.5),
                y_range=y_range,
                z_range=(z_range[0] - 2.0, z_range[1] + 5.0),
            )
            vis_xyz, vis_lab, vis_lid, _ = sag.merge_static_dynamic(
                xyz_s, lab_s, lid_s, coord, pred, lidar_ids
            )
        else:
            vis_xyz, vis_lab, vis_lid = coord, pred, lidar_ids
    else:
        vis_xyz, vis_lab, vis_lid = coord, pred, lidar_ids

    grid = occmod.build_occupancy(
        vis_xyz,
        vis_lab,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        voxel=occ_voxel,
        min_points=occ_min_points,
    )

    out_frame.mkdir(parents=True, exist_ok=True)
    cam_dir = out_frame / "cameras"
    cam_dir.mkdir(exist_ok=True)

    cameras_meta = []
    for cam_name in CAM_ORDER:
        img_path = fr / f"{cam_name}.jpg"
        if not img_path.is_file() or cam_name not in sensors:
            continue
        cam_doc = sensors[cam_name]
        K, dist5, T_c_v, T_v_c, cal_w, cal_h = parse_camera(cam_doc)
        # clean image copy (no overlays)
        dst = cam_dir / f"{cam_name}.jpg"
        shutil.copy2(img_path, dst)
        # if image size differs from calibration, note scales
        with Image.open(img_path) as im:
            iw, ih = im.size
        sx = iw / float(cal_w) if cal_w else 1.0
        sy = ih / float(cal_h) if cal_h else 1.0
        K_img = K.copy()
        K_img[0, :] *= sx
        K_img[1, :] *= sy
        cameras_meta.append(
            {
                "name": cam_name,
                "file": f"cameras/{cam_name}.jpg",
                "width": iw,
                "height": ih,
                "K": K_img.reshape(-1).tolist(),
                "dist5": dist5.tolist(),
                "T_c_v": T_c_v.reshape(-1).tolist(),
                "T_v_c": T_v_c.reshape(-1).tolist(),
            }
        )

    write_f32(out_frame / "occ_centers.f32.bin", grid.centers)
    write_u8(out_frame / "occ_labels.u8.bin", grid.labels.astype(np.uint8))
    write_i32(out_frame / "occ_ijk.i32.bin", grid.ijk)
    write_i32(out_frame / "occ_counts.i32.bin", grid.counts)

    points_info = None
    if export_points:
        n = vis_xyz.shape[0]
        if n > max_export_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=max_export_points, replace=False)
            p_xyz = vis_xyz[idx]
            p_lab = vis_lab[idx]
            p_lid = vis_lid[idx]
        else:
            p_xyz, p_lab, p_lid = vis_xyz, vis_lab, vis_lid
        write_f32(out_frame / "points_xyz.f32.bin", p_xyz)
        write_u8(out_frame / "points_labels.u8.bin", np.asarray(p_lab, dtype=np.uint8))
        write_u8(out_frame / "points_lidar_id.u8.bin", np.asarray(p_lid, dtype=np.uint8))
        points_info = {
            "n": int(p_xyz.shape[0]),
            "xyz": "points_xyz.f32.bin",
            "labels": "points_labels.u8.bin",
            "lidar_id": "points_lidar_id.u8.bin",
        }

    pose = (meta.get("dependency") or {}).get("ego_pose", {}).get("pose")
    frame_meta = {
        "timestamp": ts,
        "voxel": float(grid.voxel),
        "x_range": list(x_range),
        "y_range": list(y_range),
        "z_range": list(z_range),
        "occ_shape": list(grid.shape),
        "n_occ": int(grid.centers.shape[0]),
        "n_static_roi": int(xyz_s.shape[0]),
        "n_vis_points": int(vis_xyz.shape[0]),
        "occupancy": {
            "centers": "occ_centers.f32.bin",
            "labels": "occ_labels.u8.bin",
            "ijk": "occ_ijk.i32.bin",
            "counts": "occ_counts.i32.bin",
        },
        "points": points_info,
        "cameras": cameras_meta,
        "ego_pose": pose,
        "class_names": list(vis.WAYMO_NAMES),
        "class_colors_rgb": (vis.WAYMO_COLORS * 255.0).astype(np.uint8).tolist(),
        "vehicle_frame": "+y forward, +x right, +z up",
    }
    (out_frame / "meta.json").write_text(json.dumps(frame_meta, indent=2))
    return {
        "timestamp": ts,
        "n_occ": frame_meta["n_occ"],
        "n_cameras": len(cameras_meta),
        "path": str(out_frame),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clip",
        default="stop_1784423032302844849_vehicle-V002-20260719_090818",
    )
    ap.add_argument("--backup-root", default="data/robotruck_clips_backup")
    ap.add_argument("--out-dir", default="exp/robotruck/occ_scenes")
    ap.add_argument("--pred-dir", default="", help="Default: exp/robotruck/clip_video/<clip>/preds")
    ap.add_argument("--config-file", default="configs/waymo/semseg-litept-small-v1m1.py")
    ap.add_argument(
        "--weight",
        default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth",
    )
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse-pred", action="store_true")
    ap.add_argument("--aggregate-static", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--static-voxel", type=float, default=0.25)
    ap.add_argument("--agg-stride", type=int, default=5)
    ap.add_argument("--occ-voxel", type=float, default=0.1)
    ap.add_argument("--occ-min-points", type=int, default=1)
    ap.add_argument("--bev-x-half", type=float, default=30.0)
    ap.add_argument("--bev-y-min", type=float, default=-200.0)
    ap.add_argument("--bev-y-max", type=float, default=400.0)
    ap.add_argument("--z-min", type=float, default=-5.0)
    ap.add_argument("--z-max", type=float, default=20.0)
    ap.add_argument("--export-points", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-export-points", type=int, default=200000)
    args = ap.parse_args()

    clip_dir = (ROOT / args.backup_root / args.clip).resolve()
    if not clip_dir.is_dir():
        raise FileNotFoundError(clip_dir)
    scene_root = (ROOT / args.out_dir / args.clip).resolve()
    scene_root.mkdir(parents=True, exist_ok=True)
    pred_dir = (
        Path(args.pred_dir).resolve()
        if args.pred_dir
        else (ROOT / "exp/robotruck/clip_video" / args.clip / "preds").resolve()
    )
    pred_dir.mkdir(parents=True, exist_ok=True)

    all_ts = list_clip_frames(clip_dir)
    timestamps = all_ts[:: max(1, args.stride)]
    if args.max_frames > 0:
        timestamps = timestamps[: args.max_frames]
    print(f"export clip={args.clip} frames={len(timestamps)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = _h.load_segmentor(ROOT / args.config_file, ROOT / args.weight, device)

    x_range = (-args.bev_x_half, args.bev_x_half)
    y_range = (args.bev_y_min, args.bev_y_max)
    z_range = (args.z_min, args.z_max)

    static_agg = None
    if args.aggregate_static:
        cache_path = (
            ROOT
            / "exp/robotruck/clip_video"
            / args.clip
            / "static_agg"
            / f"static_voxel{args.static_voxel:g}_s{args.agg_stride}.npz"
        )
        agg_ts = all_ts[:: max(1, args.agg_stride)]
        static_agg = sag.load_or_build_static_aggregate(
            clip_dir,
            pred_dir,
            agg_ts,
            load_lidar_bin=_h.load_lidar_bin,
            lidar_cols=len(_h.LIDAR_COLS),
            infer_frame=_h.infer_frame if not args.reuse_pred else None,
            model=model,
            device=device,
            grid_size=args.grid_size,
            voxel=args.static_voxel,
            cache_path=cache_path,
            use_oracle_boxes=True,
        )
        if static_agg["xyz_map"].shape[0] == 0 and args.reuse_pred:
            static_agg = sag.load_or_build_static_aggregate(
                clip_dir,
                pred_dir,
                agg_ts,
                load_lidar_bin=_h.load_lidar_bin,
                lidar_cols=len(_h.LIDAR_COLS),
                infer_frame=_h.infer_frame,
                model=model,
                device=device,
                grid_size=args.grid_size,
                voxel=args.static_voxel,
                cache_path=cache_path,
                use_oracle_boxes=True,
            )
        print(f"static_agg N={static_agg['xyz_map'].shape[0]}")

    index = {
        "clip": args.clip,
        "occ_voxel": args.occ_voxel,
        "frames": [],
        "viewer_hint": "python tools/occ_viewer/serve.py --scene " + str(scene_root),
    }
    for i, ts in enumerate(timestamps):
        out_frame = scene_root / "frames" / ts
        info = export_frame(
            clip_dir=clip_dir,
            out_frame=out_frame,
            ts=ts,
            pred_dir=pred_dir,
            static_agg=static_agg,
            model=model,
            device=device,
            grid_size=args.grid_size,
            reuse_pred=args.reuse_pred,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            occ_voxel=args.occ_voxel,
            occ_min_points=args.occ_min_points,
            export_points=args.export_points,
            max_export_points=args.max_export_points,
        )
        index["frames"].append(
            {"timestamp": ts, "dir": f"frames/{ts}", "n_occ": info["n_occ"]}
        )
        print(f"  [{i+1}/{len(timestamps)}] ts={ts} n_occ={info['n_occ']}", flush=True)

    (scene_root / "index.json").write_text(json.dumps(index, indent=2))
    # convenience copy of viewer into scene (optional symlink-like copy of note)
    (scene_root / "README.txt").write_text(
        "Open viewer:\n"
        f"  cd {ROOT}\n"
        f"  .venv_smoke/bin/python tools/occ_viewer/serve.py --scene {scene_root}\n"
        "Then open the printed URL.\n"
    )
    print(f"done -> {scene_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
