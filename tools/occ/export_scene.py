"""Prod: export OCC scene packages (cameras + voxels + optional points).

python tools/occ/export_scene.py --clip <name> --stride 2 --max-frames 3 --reuse-pred --export-points
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from paths import ROOT, ensure_import_path, setup_cuda_env, load_lidar_bin, LIDAR_COLS, ensure_preds

setup_cuda_env(warmup=False)
ensure_import_path(tools=True, repo=True)

import numpy as np
from PIL import Image

import artifact_io
import occupancy as occmod
import quality_gate as qgate
import static_agg as sag
import visualize as vis

from camera_calibration import camera_names, parse_camera, camera_metadata

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

def pose_stamp_ns(ego: dict) -> int:
    stamp = ego["header"]["stamp"]
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])

def interpolate_pose_matrix(
    samples: list[tuple[int, dict]], timestamp_ns: int
) -> np.ndarray:
    return sag.interpolate_pose_matrix(samples, int(timestamp_ns))

def write_f32(path: Path, arr: np.ndarray) -> None:
    artifact_io.write_bytes(path, np.asarray(arr, dtype=np.float32).reshape(-1).tobytes())

def write_u8(path: Path, arr: np.ndarray) -> None:
    artifact_io.write_bytes(path, np.asarray(arr, dtype=np.uint8).reshape(-1).tobytes())

def write_i32(path: Path, arr: np.ndarray) -> None:
    artifact_io.write_bytes(path, np.asarray(arr, dtype=np.int32).reshape(-1).tobytes())

def asset_ref(uri: str, dtype: str, shape: list) -> dict:
    return {
        "uri": uri,
        "dtype": dtype,
        "shape": list(shape),
        "byte_order": "little",
    }


def link_reused_assets(scene_root: Path, previous_scene: Path, value):
    """Reference immutable prior assets without regenerating or publishing them."""
    if isinstance(value, dict):
        if 'uri' in value:
            relative = Path(value['uri'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Expected scene-relative asset: {relative}')
            source = (previous_scene / relative).resolve(strict=True)
            target = scene_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if target.resolve() != source:
                    raise ValueError(f'Reused asset conflicts with existing file: {target}')
            else:
                target.symlink_to(source)
        else:
            for child in value.values():
                link_reused_assets(scene_root, previous_scene, child)
    elif isinstance(value, list):
        for child in value:
            link_reused_assets(scene_root, previous_scene, child)

def coarse_taxonomy() -> dict:
    # 0 dynamic, 1 static, 2 freespace, 3 noise
    fine_to_coarse = []
    dynamic = {0, 1, 2, 3, 4, 5, 6, 11, 12}
    freespace = {17, 18, 19, 20, 21}
    static = {7, 8, 9, 10, 13, 14, 15, 16}
    for i in range(22):
        if i in dynamic:
            fine_to_coarse.append(0)
        elif i in static:
            fine_to_coarse.append(1)
        elif i in freespace:
            fine_to_coarse.append(2)
        else:
            fine_to_coarse.append(3)
    return {
        "names": ["dynamic", "static", "freespace", "noise"],
        "colors_rgb": [[230, 64, 64], [64, 160, 255], [72, 200, 96], [160, 160, 160]],
        "fine_to_coarse": fine_to_coarse,
    }

def export_frame(
    *,
    clip_id: str,
    clip_dir: Path,
    scene_root: Path,
    out_frame: Path,
    ts: str,
    pred_dir: Path,
    static_agg: dict | None,
    grid_size: float,
    reuse_pred: bool,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    occ_voxel: float,
    occ_min_points: int,
    export_points: bool,
    max_export_points: int,
    ego_filter: dict | None,
    pose_samples: list[tuple[int, dict]],
    geometry: dict | None = None,
    camera_executor: ThreadPoolExecutor | None = None,
    reuse_scene: Path | None = None,
    shared_static: bool = False,
) -> dict:
    if max_export_points != 0:
        raise ValueError('Point-cloud downsampling is disabled; point cap must be zero')
    fr = clip_dir / "frames" / ts
    meta = json.loads((fr / "frame.json").read_text())
    sensors = meta["dependency"]["sensors"]
    lidar_timestamp = sag.lidar_reference_timestamp(meta, ts)
    reference_pose = sag.interpolate_pose(pose_samples, lidar_timestamp)
    T_map_v_lidar = sag.ego_pose_to_T_map_vehicle(reference_pose)

    pts = load_lidar_bin(fr / "lidar_merge.bin")
    coord = pts[:, :3].astype(np.float32)
    lidar_ids = pts[:, 6].astype(np.int32)
    intensity = pts[:, 3]

    pred_path = pred_dir / f"{ts}_pred.npy"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"missing pred {pred_path}; run A warmup CLI or export with ensure_preds"
        )
    pred = np.load(pred_path).astype(np.int64).reshape(-1)
    if pred.shape[0] != coord.shape[0]:
        raise ValueError(
            f"pred length {pred.shape[0]} != lidar N {coord.shape[0]} for {ts}"
        )

    # Preserve the complete current-frame deskew cloud for camera calibration
    # inspection. OCC ego/ROI/static aggregation filters must not affect it.
    frame_sensor_xyz = coord.copy()
    frame_sensor_labels = pred.astype(np.int32, copy=True)
    frame_sensor_lidar_ids = lidar_ids.copy()
    reused = None
    if reuse_scene is not None:
        previous = json.loads((reuse_scene / 'frames' / ts / 'meta.json').read_text())
        if previous.get('geometry') != (geometry or {}) or int(previous['ego_pose_timestamp_ns']) != lidar_timestamp:
            raise ValueError('Cannot reuse assets across different geometry inputs')
        reused = previous['assets']
        if reused['frame_sensor_points']['n'] != len(coord):
            raise ValueError('Cached single-frame cloud has a different point count')
        for group in ('cameras', 'frame_sensor_points'):
            link_reused_assets(scene_root, reuse_scene, reused[group])

    ego_keep, ego_filter_stats = sag.ground_aware_ego_keep_mask(coord, pred, ego_filter)
    coord, pred, lidar_ids = coord[ego_keep], pred[ego_keep], lidar_ids[ego_keep]

    lab_s = np.zeros((0,), np.int32)
    lid_s = np.zeros((0,), np.int32)
    xyz_s = np.zeros((0, 3), np.float32)
    if static_agg is not None and static_agg["xyz_map"].shape[0] > 0:
        pose = reference_pose
        if pose:
            T_map_v = T_map_v_lidar
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

    def export_camera(cam_name):
        img_path = fr / f"{cam_name}.jpg"
        if not img_path.is_file() or cam_name not in sensors:
            return None
        cam_doc = sensors[cam_name]
        # Camera corruption must not invalidate a LiDAR OCC frame.  Verify the
        # source before copying and omit only the unreadable camera asset.
        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
                im.verify()
        except (OSError, ValueError) as error:
            print(f"  [camera-skip] ts={ts} camera={cam_name}: {error}", flush=True)
            return None
        # clean image copy (no overlays)
        dst = cam_dir / f"{cam_name}.jpg"
        # The source cache is immutable. A hard link preserves the exact JPEG
        # without copying its bytes again; cross-filesystem exports still work.
        try:
            if dst.exists():
                dst.unlink()
            os.link(img_path.resolve(), dst)
        except OSError:
            shutil.copy2(img_path, dst)
        artifact_io.camera_ready(dst, cam_doc.get("md5"))
        return camera_metadata(
            cam_name, cam_doc, (iw, ih), f"frames/{ts}/cameras/{cam_name}.jpg",
            lidar_timestamp, pose_samples, geometry=geometry)

    camera_map = camera_executor.map if camera_executor is not None else map
    cameras_meta = (reused['cameras'] if reused is not None else
                    [c for c in camera_map(export_camera, camera_names(sensors)) if c is not None])

    prefix = f"frames/{ts}"
    write_f32(out_frame / "occ_centers.f32.bin", grid.centers)
    write_u8(out_frame / "occ_labels.u8.bin", grid.labels.astype(np.uint8))
    write_i32(out_frame / "occ_ijk.i32.bin", grid.ijk)
    write_i32(out_frame / "occ_counts.i32.bin", grid.counts)

    n_occ = int(grid.centers.shape[0])
    points_info = None
    points_assets = None
    frame_sensor_points_assets = None
    n_points_exported = 0
    if export_points and reused is not None:
        frame_sensor_points_assets = reused['frame_sensor_points']
    if export_points and reused is None:
        write_f32(out_frame / "frame_sensor_points_xyz.f32.bin", frame_sensor_xyz)
        write_u8(
            out_frame / "frame_sensor_points_labels.u8.bin",
            np.asarray(frame_sensor_labels, dtype=np.uint8),
        )
        write_u8(
            out_frame / "frame_sensor_points_lidar_id.u8.bin",
            np.asarray(frame_sensor_lidar_ids, dtype=np.uint8),
        )
        n_frame_sensor = int(frame_sensor_xyz.shape[0])
        frame_sensor_points_assets = {
            "n": n_frame_sensor,
            "source": (geometry or {}).get('point_processing', 'lidar_merge_deskew'),
            "filtering": "none",
            "xyz": asset_ref(
                f"{prefix}/frame_sensor_points_xyz.f32.bin",
                "float32",
                [n_frame_sensor, 3],
            ),
            "labels": asset_ref(
                f"{prefix}/frame_sensor_points_labels.u8.bin",
                "uint8",
                [n_frame_sensor],
            ),
            "lidar_id": asset_ref(
                f"{prefix}/frame_sensor_points_lidar_id.u8.bin",
                "uint8",
                [n_frame_sensor],
            ),
        }
    if export_points:
        # Preserve every selected static and current-frame dynamic point.
        composed = shared_static and static_agg is not None and len(static_agg['xyz_map']) > 0
        if composed:
            dyn = np.isin(pred, list(sag.WAYMO_DYNAMIC_IDS))
            p_xyz, p_lab, p_lid = coord[dyn], pred[dyn], lidar_ids[dyn]
        elif xyz_s.shape[0] > 0:
            # re-merge explicitly for stable export ordering: static then dyn
            dyn = np.isin(pred.astype(np.int64), list(sag.WAYMO_DYNAMIC_IDS))
            xyz_d = coord[dyn]
            lab_d = pred[dyn].astype(np.int32)
            lid_d = lidar_ids[dyn]
            n_d = int(xyz_d.shape[0])
            n_s = int(xyz_s.shape[0])
            p_xyz = np.concatenate([xyz_s, xyz_d], axis=0) if n_d else xyz_s
            p_lab = np.concatenate([lab_s, lab_d], axis=0) if n_d else lab_s
            p_lid = np.concatenate([lid_s, lid_d], axis=0) if n_d else lid_s
        else:
            p_xyz, p_lab, p_lid = vis_xyz, vis_lab, vis_lid
        write_f32(out_frame / "points_xyz.f32.bin", p_xyz)
        write_u8(out_frame / "points_labels.u8.bin", np.asarray(p_lab, dtype=np.uint8))
        write_u8(out_frame / "points_lidar_id.u8.bin", np.asarray(p_lid, dtype=np.uint8))
        n_points_exported = int(p_xyz.shape[0])
        points_info = {
            "n": n_points_exported,
            "xyz": "points_xyz.f32.bin",
            "labels": "points_labels.u8.bin",
            "lidar_id": "points_lidar_id.u8.bin",
        }
        points_assets = {
            "n": n_points_exported,
            "xyz": asset_ref(f"{prefix}/points_xyz.f32.bin", "float32", [n_points_exported, 3]),
            "labels": asset_ref(f"{prefix}/points_labels.u8.bin", "uint8", [n_points_exported]),
            "lidar_id": asset_ref(
                f"{prefix}/points_lidar_id.u8.bin", "uint8", [n_points_exported]
            ),
        }
        if composed:
            points_assets.update({
                'layout': 'clip_static_plus_frame_dynamic/v1',
                'static_ref': 'index.json#static_agg',
                'logical_n': int(vis_xyz.shape[0]),
                'static_n': int(xyz_s.shape[0]),
                'T_vehicle_map': np.linalg.inv(T_map_v_lidar).reshape(-1).tolist(),
                'static_roi': {'x': [x_range[0] * 1.5, x_range[1] * 1.5],
                               'y': list(y_range), 'z': [z_range[0] - 2.0, z_range[1] + 5.0]},
                'static_roi_inclusive': True,
                'ordering': 'static_then_dynamic',
                'point_downsampling': False,
            })

    pose = reference_pose
    origin = [float(x_range[0]), float(y_range[0]), float(z_range[0])]
    frame_meta = {
        "schema_version": "robotruck_occ_frame/v1",
        "scene_id": clip_id,
        "frame_id": ts,
        "timestamp": ts,
        "coordinate": {
            "frame": "vehicle",
            "x": "lateral",
            "y": "forward",
            "z": "up",
        },
        "grid": {
            "voxel": float(grid.voxel),
            "x_range": list(x_range),
            "y_range": list(y_range),
            "z_range": list(z_range),
            "shape": list(grid.shape),
            "origin": origin,
            "index_rule": (
                "ijk = floor((p - origin) / voxel); "
                "center = origin + (ijk + 0.5) * voxel"
            ),
        },
        "stats": {
            "n_occ": n_occ,
            "n_static_roi": int(xyz_s.shape[0]),
            "n_vis_points": int(vis_xyz.shape[0]),
            "n_points_exported": n_points_exported,
            "ego_filter_removed": int(ego_filter_stats["removed"]),
        },
        "ego_filter": {"config": ego_filter, "stats": ego_filter_stats},
        "assets": {
            "occupancy": {
                "n": n_occ,
                "ijk": asset_ref(f"{prefix}/occ_ijk.i32.bin", "int32", [n_occ, 3]),
                "labels": asset_ref(f"{prefix}/occ_labels.u8.bin", "uint8", [n_occ]),
                "centers": asset_ref(
                    f"{prefix}/occ_centers.f32.bin", "float32", [n_occ, 3]
                ),
                "counts": asset_ref(f"{prefix}/occ_counts.i32.bin", "int32", [n_occ]),
            },
            "points": points_assets,
            "frame_sensor_points": frame_sensor_points_assets,
            "cameras": cameras_meta,
        },
        "ego_pose": pose,
        "ego_pose_timestamp_ns": str(lidar_timestamp),
        "pose_reference": "lidar_timestamp",
        "geometry": geometry or {},
        "taxonomy_ref": "index.json#taxonomy",
        # ---- legacy fields (viewer / tools backward compatible) ----
        "voxel": float(grid.voxel),
        "x_range": list(x_range),
        "y_range": list(y_range),
        "z_range": list(z_range),
        "occ_shape": list(grid.shape),
        "n_occ": n_occ,
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
        "class_names": list(vis.WAYMO_NAMES),
        "class_colors_rgb": (vis.WAYMO_COLORS * 255.0).astype(np.uint8).tolist(),
        "vehicle_frame": "+y forward, +x lateral, +z up",
    }
    (out_frame / "meta.json").write_text(json.dumps(frame_meta, indent=2))
    return {
        "timestamp": ts,
        "frame_id": ts,
        "n_occ": n_occ,
        "n_points": n_points_exported,
        "n_cameras": len(cameras_meta),
        "meta_uri": f"{prefix}/meta.json",
        "path": str(out_frame),
    }

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="")
    ap.add_argument("--clips-json", default="", help="JSON list of {clip_id}; overrides --clip")
    ap.add_argument("--scene-prefix", default="", help="scene/cache name = prefix + clip_id")
    ap.add_argument("--backup-root", default="data/robotruck_clips_backup")
    ap.add_argument("--out-dir", default="exp/robotruck/occ_scenes")
    ap.add_argument("--timings-dir", default="", help="Optional per-clip timing sidecars outside Scene assets")
    ap.add_argument("--pred-dir", default="", help="Default: exp/robotruck/clip_video/<clip>/preds")
    ap.add_argument("--config-file", default="configs/waymo/semseg-litept-small-v1m1.py")
    ap.add_argument(
        "--weight",
        default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth",
    )
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=3)
    ap.add_argument("--skip-initial-seconds", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse-pred", action="store_true")
    ap.add_argument("--require-cached-preds", action="store_true",
                    help="Fail on missing predictions instead of starting inference")
    ap.add_argument("--reuse-scene", type=Path,
                    help="Reuse existing cameras/single-frame clouds; regenerate only aggregation and OCC")
    ap.add_argument("--aggregate-static", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--static-voxel", type=float, default=0.0,
                    help="Deprecated compatibility option; static point downsampling is disabled")
    ap.add_argument("--agg-stride", type=int, default=1,
                    help="Must be 1: aggregate all source frames")
    ap.add_argument("--occ-voxel", type=float, default=0.25)
    ap.add_argument("--occ-min-points", type=int, default=1)
    ap.add_argument("--bev-x-half", type=float, default=30.0)
    ap.add_argument("--bev-y-min", type=float, default=-200.0)
    ap.add_argument("--bev-y-max", type=float, default=400.0)
    ap.add_argument("--z-min", type=float, default=-5.0)
    ap.add_argument("--z-max", type=float, default=20.0)
    ap.add_argument("--export-points", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-export-points", type=int, default=0,
                    help="0 exports every point; positive values are rejected to prevent downsampling")
    ap.add_argument("--camera-workers", type=int, default=4)
    ap.add_argument('--frame-workers', type=int, default=1)
    ap.add_argument('--shared-static', action='store_true',
                    help='Store clip static once and only dynamic points per frame (requires compatible reader)')
    ap.add_argument('--no-static-cache', action='store_true', help='Avoid compressing a transient aggregate cache')
    ap.add_argument("--geometry-quality-gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--layer-threshold", type=float, default=0.15)
    ap.add_argument("--pose-shift-threshold", type=float, default=0.40)
    ap.add_argument("--quality-sample-frames", type=int, default=5)
    ap.add_argument("--ego-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ego-x-range", type=float, nargs=2, default=(-1.65, 1.65))
    ap.add_argument("--ego-y-range", type=float, nargs=2, default=(-1.0, 2.5))
    ap.add_argument("--ego-min-height", type=float, default=0.35)
    ap.add_argument("--ego-max-height", type=float, default=4.0)
    args = ap.parse_args(argv)
    if args.agg_stride != 1:
        ap.error('Static aggregation must use every frame; --agg-stride must be 1')
    if args.frame_workers < 1:
        ap.error('--frame-workers must be positive')
    if args.shared_static and (not args.aggregate_static or not args.export_points):
        ap.error('--shared-static requires static aggregation and exported points')
    if args.require_cached_preds and not args.reuse_pred:
        ap.error('--require-cached-preds requires --reuse-pred')
    if args.reuse_scene and (not args.require_cached_preds or args.stride != 1 or args.max_frames != 0 or args.skip_initial_seconds != 0):
        ap.error('--reuse-scene requires cached predictions, stride=1, max-frames=0 and skip-initial-seconds=0')
    if args.static_voxel != 0:
        ap.error("static point downsampling is disabled; --static-voxel must be 0")
    if args.max_export_points != 0:
        ap.error("point-cloud downsampling is disabled; --max-export-points must be 0")

    clips: list[str]
    if args.clips_json:
        payload = json.loads(Path(args.clips_json).read_text())
        clips = [c["clip_id"] if isinstance(c, dict) else str(c) for c in payload]
    elif args.clip:
        clips = [args.clip]
    else:
        raise SystemExit("need --clip or --clips-json")

    rc = 0
    for clip_id in clips:
        scene = f"{args.scene_prefix}{clip_id}" if args.scene_prefix else clip_id
        one = argparse.Namespace(**vars(args))
        one.clip = scene
        try:
            rc = max(rc, int(export_one(one) or 0))
        except Exception as exc:
            print(f"FAIL {scene}: {exc}", flush=True)
            rc = max(rc, 1)
    return rc

def export_one(args) -> int:
    timings = {}
    clip_dir = (ROOT / args.backup_root / args.clip).resolve()
    if not clip_dir.is_dir():
        raise FileNotFoundError(clip_dir)
    scene_root = (ROOT / args.out_dir / args.clip).resolve()
    pred_dir = (
        Path(args.pred_dir).resolve()
        if args.pred_dir
        else (ROOT / "exp/robotruck/clip_video" / args.clip / "preds").resolve()
    )

    all_ts = list_clip_frames(clip_dir)
    if len(set(all_ts)) != len(all_ts):
        raise ValueError('Duplicate source timestamps')
    from pose_trajectory import load_geometry
    geometry = load_geometry(clip_dir)
    pose_samples = sag.load_pose_samples(clip_dir)
    if len(pose_samples) < 2:
        raise ValueError("at least two timestamped ego poses are required")
    export_ts = list(all_ts)
    if export_ts and args.skip_initial_seconds > 0:
        cutoff = int(all_ts[0]) + int(args.skip_initial_seconds * 1e9)
        export_ts = [ts for ts in export_ts if int(ts) >= cutoff]
        print(f"skip initial {args.skip_initial_seconds:g}s -> first={export_ts[0] if export_ts else 'none'}")
    timestamps = export_ts[:: max(1, args.stride)]
    if args.max_frames > 0:
        timestamps = timestamps[: args.max_frames]
    reuse_scene = getattr(args, 'reuse_scene', None)
    if reuse_scene is not None:
        reuse_scene = reuse_scene.resolve(strict=True)
        if reuse_scene == scene_root:
            raise ValueError('Derived refresh must use a new scene directory')
        old_index = json.loads((reuse_scene / 'index.json').read_text())
        if old_index.get('geometry') != geometry:
            raise ValueError('Previous scene geometry differs from selected inputs')
        timestamps = [str(row['timestamp']) for row in old_index['frames']]
        if not set(timestamps) <= set(all_ts):
            raise ValueError('Previous scene has unknown source timestamps')
    stage_started = time.monotonic()
    geometry_quality = qgate.assess_clip_geometry(
        clip_dir,
        all_ts,
        sample_frames=args.quality_sample_frames,
        layer_threshold=args.layer_threshold,
        pose_shift_threshold=args.pose_shift_threshold,
        pose_samples=pose_samples,
    )
    timings["geometry_quality_seconds"] = time.monotonic() - stage_started
    if args.geometry_quality_gate and not geometry_quality["allow_occ"]:
        geometry_quality["action"] = "clip_rejected"
        raise RuntimeError("GEOMETRY_QUALITY_REJECTED: " + json.dumps(geometry_quality, sort_keys=True))
    else:
        geometry_quality["action"] = "occ_allowed"
    scene_root.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"export clip={args.clip} frames={len(timestamps)}")

    x_range = (-args.bev_x_half, args.bev_x_half)
    y_range = (args.bev_y_min, args.bev_y_max)
    z_range = (args.z_min, args.z_max)
    ego_filter = {
        "enabled": bool(args.ego_filter),
        "x_range": list(args.ego_x_range),
        "y_range": list(args.ego_y_range),
        "min_height": args.ego_min_height,
        "max_height": args.ego_max_height,
        "ground_fit_margin": 0.5,
        "method": "semantic_ground_robust_plane/v1",
    }

    if args.agg_stride != 1:
        raise ValueError('Static aggregation must use every source frame')
    agg_ts = list(all_ts) if args.aggregate_static else []
    # B never loads the model: ensure preds via A CLI, then read npy only.
    if not args.reuse_pred:
        for ts in list(dict.fromkeys(list(timestamps) + list(agg_ts))):
            pp = pred_dir / f"{ts}_pred.npy"
            if pp.is_file():
                pp.unlink()
    stage_started = time.monotonic()
    ensure_preds(
        clip_dir=clip_dir,
        pred_dir=pred_dir,
        timestamps=timestamps,
        agg_timestamps=agg_ts,
        stride=args.stride,
        agg_stride=args.agg_stride,
        max_frames=args.max_frames if args.max_frames > 0 else len(timestamps),
        config_file=args.config_file,
        weight=args.weight,
        grid_size=args.grid_size,
        require_cached=getattr(args, 'require_cached_preds', False),
    )
    timings["prediction_wait_and_run_seconds"] = time.monotonic() - stage_started

    stage_started = time.monotonic()
    static_agg = None
    if args.aggregate_static:
        cache_path = (
            ROOT
            / "exp/robotruck/clip_video"
            / args.clip
            / "static_agg"
            / f"static_voxel{args.static_voxel:g}_s{args.agg_stride}.npz"
        )
        static_agg = sag.load_or_build_static_aggregate(
            clip_dir,
            pred_dir,
            agg_ts,
            load_lidar_bin=load_lidar_bin,
            lidar_cols=len(LIDAR_COLS),
            infer_frame=None,
            model=None,
            device=None,
            grid_size=args.grid_size,
            voxel=0.0,
            cache_path=None if getattr(args, 'no_static_cache', False) else cache_path,
            use_oracle_boxes=True,
            ego_filter=ego_filter,
            pose_samples=pose_samples,
        )
        print(f"static_agg N={static_agg['xyz_map'].shape[0]}")
        # Mirror static_agg into the scene package (map frame; frames only apply pose).
        n_sa = int(static_agg["xyz_map"].shape[0])
        if n_sa == 0:
            index_static_agg = None
        else:
            sa_dir = scene_root / "static_agg"
            sa_dir.mkdir(parents=True, exist_ok=True)
            xyz_m = static_agg["xyz_map"]
            lab_m = static_agg["labels"].astype(np.uint8)
            lid_m = static_agg["lidar_ids"].astype(np.uint8)
            write_f32(sa_dir / "xyz_map.f32.bin", xyz_m)
            write_u8(sa_dir / "labels.u8.bin", lab_m)
            write_u8(sa_dir / "lidar_id.u8.bin", lid_m)
            index_static_agg = {
                "voxel": None,
                "n": n_sa,
                "xyz_map": {
                    "uri": "static_agg/xyz_map.f32.bin",
                    "dtype": "float32",
                    "shape": [n_sa, 3],
                },
                "labels": {
                    "uri": "static_agg/labels.u8.bin",
                    "dtype": "uint8",
                    "shape": [n_sa],
                },
                "lidar_id": {
                    "uri": "static_agg/lidar_id.u8.bin",
                    "dtype": "uint8",
                    "shape": [n_sa],
                },
                "note": "Clip-level static in map frame; each frame only applies ego_pose.",
            }
    else:
        index_static_agg = None
    timings["static_aggregate_seconds"] = time.monotonic() - stage_started

    index = {
        "schema_version": "robotruck_occ_scene/v1",
        "scene_id": args.clip,
        "clip_id": args.clip,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "geometry_quality": geometry_quality,
        "geometry": geometry,
        "aggregation": {
            "stride": 1, "source_timestamps": agg_ts,
            "source_frame_count": len(agg_ts), "point_downsampling": False,
        },
        "defaults": {
            "occ_voxel": args.occ_voxel,
            "roi": {"x": [-24.0, 24.0], "y": [-25.0, 150.0], "z": [-5.0, 3.0]},
            "vehicle_frame": {
                "x": "lateral",
                "y": "forward",
                "z": "up",
            },
            "ego_filter": ego_filter,
        },
        "taxonomy": {
            "fine": {
                "n": len(vis.WAYMO_NAMES),
                "names": list(vis.WAYMO_NAMES),
                "colors_rgb": (vis.WAYMO_COLORS * 255.0).astype(np.uint8).tolist(),
            },
            "coarse": coarse_taxonomy(),
            "lidar_ids": {
                "ids": [1, 2, 14],
                "colors_rgb": [[40, 180, 255], [40, 220, 40], [255, 40, 40]],
            },
        },
        "frames": [],
        "clip": args.clip,
        "occ_voxel": args.occ_voxel,
        "viewer_hint": "cd /home/dev/01develop/occ_viewer && python serve.py --scene " + str(scene_root),
    }
    if index_static_agg is not None:
        index["static_agg"] = index_static_agg
    # Initialize the immutable transform input once before concurrent frame work.
    if static_agg is not None and getattr(args, 'frame_workers', 1) > 1:
        sag.prepare_static_transform(static_agg)
    stage_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.camera_workers)) as camera_executor:
        def one_frame(ts):
            started = time.monotonic()
            out_frame = scene_root / "frames" / ts
            info = export_frame(
                clip_id=args.clip,
                clip_dir=clip_dir,
                scene_root=scene_root,
                out_frame=out_frame,
                ts=ts,
                pred_dir=pred_dir,
                static_agg=static_agg,
                grid_size=args.grid_size,
                reuse_pred=args.reuse_pred,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                occ_voxel=args.occ_voxel,
                occ_min_points=args.occ_min_points,
                export_points=args.export_points,
                max_export_points=args.max_export_points,
                ego_filter=ego_filter,
                pose_samples=pose_samples,
                geometry=geometry,
                camera_executor=camera_executor,
                reuse_scene=reuse_scene,
                shared_static=getattr(args, 'shared_static', False),
            )
            print(f'frame ready ts={ts} seconds={time.monotonic() - started:.2f} n_occ={info["n_occ"]}', flush=True)
            return info
        with ThreadPoolExecutor(max_workers=getattr(args, 'frame_workers', 1)) as frame_executor:
            infos = list(frame_executor.map(one_frame, timestamps))
        for i, (ts, info) in enumerate(zip(timestamps, infos)):
            index["frames"].append(
                {
                    "frame_id": info["frame_id"],
                    "timestamp": ts,
                    "meta_uri": info["meta_uri"],
                    "dir": f"frames/{ts}",
                    "n_occ": info["n_occ"],
                    "n_points": info["n_points"],
                }
            )
            print(f"  [{i+1}/{len(timestamps)}] ts={ts} n_occ={info['n_occ']}", flush=True)

    timings["frame_export_seconds"] = time.monotonic() - stage_started
    (scene_root / "index.json").write_text(json.dumps(index, indent=2))
    if getattr(args, "timings_dir", ""):
        timing_path = Path(args.timings_dir) / f"{args.clip}.timings.json"
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(timings, indent=2))
    print(f"done -> {scene_root}  viewer: cd /home/dev/01develop/occ_viewer && python serve.py --scene {scene_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
