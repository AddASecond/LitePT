#!/usr/bin/env python3
"""In-process OCC export+store runner (HAMI-safe single CUDA process).

Production quality: each clip runs robotruck_quality_gate before OCC export
(same gate as export_robotruck_occ_scene). Rejected clips are skipped.

HAMI notes:
  * source LitePT/.cuda_env.sh before launching Python
  * keep all CUDA work in this one interpreter (no CUDA subprocesses)

Usage (from LitePT root):
  source .cuda_env.sh
  .venv_smoke/bin/python tools/run_random10_inproc.py \\
      --clips-json /tmp/random10_clips.json \\
      --backup-root exp/robotruck/raw_volume_cache \\
      --scenes-root exp/robotruck/occ_scenes \\
      --asset-root exp/robotruck/occ_assets \\
      --stride 5 [--dry-run] [--max-frames-per-clip 60]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
assert ROOT.name == "LitePT", "run from LitePT checkout; got: " + str(ROOT)


def load_mod(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-json", required=True, type=Path)
    ap.add_argument("--backup-root", required=True, type=Path)
    ap.add_argument("--scenes-root", required=True, type=Path)
    ap.add_argument("--asset-root", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames-per-clip", type=int, default=0,
                    help="0 = all frames of the clip subset (stride filtered)")
    ap.add_argument("--aggregate-static", action="store_true", default=True)
    ap.add_argument("--no-aggregate-static", dest="aggregate_static", action="store_false")
    ap.add_argument("--export-points", action="store_true", default=True)
    ap.add_argument("--occ-voxel", type=float, default=0.2)
    ap.add_argument("--occ-min-points", type=int, default=2)
    ap.add_argument("--max-export-points", type=int, default=65536)
    ap.add_argument("--raw-frame-collection", default="raw_data_frames_lidar14_0813")
    ap.add_argument("--raw-clip-collection", default="raw_data_clips_lidar14_0813")
    ap.add_argument("--write-store", action="store_true", default=True,
                    help="write assets + mongo doc (default True).")
    ap.add_argument("--no-write-store", dest="write_store", action="store_false",
                    help="disable store step (useful for smoke tests).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--geometry-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject clips that fail robotruck_quality_gate (default: on)",
    )
    ap.add_argument("--quality-sample-frames", type=int, default=5)
    ap.add_argument("--layer-threshold", type=float, default=None)
    ap.add_argument("--pose-shift-threshold", type=float, default=None)
    args = ap.parse_args()

    clips = json.loads(args.clips_json.read_text())
    print(f"[plan] clips={len(clips)} stride={args.stride} scenes_root={args.scenes_root}")
    print(f"[plan] aggregate_static={args.aggregate_static} export_points={args.export_points}")
    print(f"[plan] write_store={args.write_store} dry_run={args.dry_run}")
    print(f"[plan] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"_CUDA_COMPAT_PATH={os.environ.get('_CUDA_COMPAT_PATH')}")

    # ------------------------------------------------------------------
    # Import project modules (runs their module-level _setup_cuda_env).
    # `export_scene` module re-exports everything we need.
    export_mod = load_mod("export_scene", "tools/export_robotruck_occ_scene.py")
    qgate = load_mod("robotruck_quality_gate", "tools/robotruck_quality_gate.py")
    list_clip_frames = export_mod.list_clip_frames
    export_frame_fn = export_mod.export_frame
    pose_stamp_ns = export_mod.pose_stamp_ns
    sag = load_mod("static_agg", "tools/robotruck_static_agg.py")
    infer_mod = load_mod("infer_rf", "tools/infer_robotruck_mongo_frame.py")
    load_segmentor = infer_mod.load_segmentor
    store_mod = load_mod("store_gridfs", "tools/store_robotruck_occ_gridfs.py")

    import torch
    gpu_ok = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_ok else "cpu")
    print(f"[cuda] device={device}  name=", end="")
    if gpu_ok:
        print(torch.cuda.get_device_name(0))
        # Burn a trivial spconv forward now to warm up indice kernels
        import spconv.pytorch as spc
        import itertools as it
        idx = torch.zeros(2000, 4, dtype=torch.int32, device=device)
        for k, (a, b, c0) in enumerate(it.product(range(10), repeat=3)):
            if k >= 2000: break
            idx[k, 1:] = torch.tensor([a, b, c0])
        idx = idx[:k]
        idx = torch.unique(idx, dim=0)
        f = torch.randn(len(idx), 4, device=device)
        t = spc.SparseConvTensor(f, idx, [10, 10, 10], 1)
        cv = spc.SubMConv3d(4, 8, 3, indice_key="warmup").to(device)
        o = cv(t)
        torch.cuda.synchronize()
        print(f"[cuda] spconv warmup OK out={tuple(o.features.shape)}")
    else:
        print("<cpu-only>")
        # CPU fallback: load_segmentor still tries model.to(cuda) – must patch.
        # LitePT segmentor uses spconv which is CUDA-only; no fallback exists.
        raise RuntimeError("CUDA unavailable (HAMI denied). Source .cuda_env.sh "
                           "and wait for HAMI vGPU cooldown, then rerun.")

    # Shared segmentor across clips – saves reload time.
    model = None
    # Shared client handle
    mongo_client = None  # store_mod.main() uses its own MongoClient internally

    status = []
    for i, c in enumerate(clips, 1):
        clip_id = c["clip_id"]
        tag = c.get("tag", "")
        scene_name = f"batch_s5_{clip_id}"
        clip_dir = (args.backup_root / scene_name).resolve()
        out_scene = (args.scenes_root / scene_name).resolve()
        print(f"\n==== [{i}/{len(clips)}] {clip_id}  tag={tag} ====")
        if not clip_dir.is_dir():
            print(f"  SKIP: cache dir missing ({clip_dir}); run materialize pipeline first.")
            status.append({"i": i, "clip_id": clip_id, "rc": -1, "msg": "no cache dir"})
            continue

        # ---- Build pose_samples ONCE per clip (read every frame's ego_pose). ----
        all_ts = list_clip_frames(clip_dir)
        pose_samples: list[tuple[int, dict]] = []
        for ts in all_ts:
            p = clip_dir / "frames" / ts / "frame.json"
            if not p.is_file():
                continue
            meta = json.loads(p.read_text())
            ego = (meta.get("dependency") or {}).get("ego_pose")
            if ego and ego.get("pose") and ego.get("header", {}).get("stamp"):
                pose_samples.append((pose_stamp_ns(ego), ego["pose"]))
        pose_samples.sort(key=lambda r: r[0])
        if len(pose_samples) < 2:
            print("  SKIP: <2 ego poses, cannot interpolate")
            status.append({"i": i, "clip_id": clip_id, "rc": -2, "msg": "<2 poses"})
            continue
        frame_ids = all_ts[::args.stride]
        if args.max_frames_per_clip > 0:
            frame_ids = frame_ids[:args.max_frames_per_clip]
        print(f"  frames to export: {len(frame_ids)} / total {len(all_ts)} "
              f"(pose_samples={len(pose_samples)})")

        gate_kw = {"sample_frames": args.quality_sample_frames}
        if args.layer_threshold is not None:
            gate_kw["layer_threshold"] = args.layer_threshold
        if args.pose_shift_threshold is not None:
            gate_kw["pose_shift_threshold"] = args.pose_shift_threshold
        geometry_quality = qgate.assess_clip_geometry(clip_dir, all_ts, **gate_kw)
        if args.geometry_quality_gate and not geometry_quality["allow_occ"]:
            geometry_quality["action"] = "clip_rejected"
            print(f"  SKIP GEOMETRY_QUALITY_REJECTED: {geometry_quality.get('reasons')}")
            status.append({
                "i": i,
                "clip_id": clip_id,
                "rc": 2,
                "msg": "GEOMETRY_QUALITY_REJECTED",
                "geometry_quality": geometry_quality,
            })
            continue
        geometry_quality["action"] = "occ_allowed"
        print(f"  geometry_quality: allow_occ=True warnings={geometry_quality.get('warnings')}")

        # Pred cache under scene dir for reuse between runs
        pred_dir = out_scene / "_pred_cache"
        if not args.dry_run:
            pred_dir.mkdir(parents=True, exist_ok=True)

        if model is None:
            cfg = ROOT / "configs/waymo/semseg-litept-small-v1m1.py"
            ckpt = ROOT / "checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth"
            model, _ = load_segmentor(cfg, ckpt, device)
            print(f"  model loaded cfg={cfg.name} ckpt={ckpt.name}")

        # ---- clip-level static aggregate (build on all timestamps, like export tool) ----
        static_agg = None
        if args.aggregate_static and not args.dry_run:
            try:
                agg_ts = all_ts  # use all clip timestamps for denser static map
                cache_path = (out_scene / "static_agg.pkl")
                # reuse_pred: only run infer_frame when pred file missing or shape mismatch.
                # sag.load_or_build_static_aggregate calls infer_frame(model, coord, strength, device, grid_size)
                # (5 positional), so we hand it the real function directly instead of wrapping a 2-arg lambda.
                static_agg = sag.load_or_build_static_aggregate(
                    clip_dir,
                    pred_dir,
                    agg_ts,
                    load_lidar_bin=infer_mod.load_lidar_bin,
                    lidar_cols=len(infer_mod.LIDAR_COLS),
                    infer_frame=infer_mod.infer_frame,
                    model=model,
                    device=device,
                    grid_size=0.1,
                    voxel=args.occ_voxel,
                    cache_path=cache_path,
                    use_oracle_boxes=True,
                    ego_filter={"enabled": True,
                                 "x_range": (-3.6, 3.6), "y_range": (-1.2, 1.2),
                                 "min_height": 0.35, "max_height": 4.0,
                                 "ground_fit_margin": 0.5},
                    require_deskew=True,
                )
                sxyz_n = static_agg["xyz_map"].shape[0] if static_agg else 0
                print(f"  static_agg: {sxyz_n} points")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  WARN static_agg failed: {e}")

        x_range = (-50.0, 50.0)
        y_range = (-50.0, 50.0)
        z_range = (-3.0, 10.0)

        # ---- per-frame export (inline) ----
        frame_paths = []
        last_frame_meta = None
        for fid in frame_ids:
            if args.dry_run:
                print(f"    dry-run export_frame {fid}")
                continue
            try:
                frame_doc = export_frame_fn(
                    clip_id=clip_id, clip_dir=clip_dir,
                    scene_root=out_scene,
                    out_frame=out_scene / "frames" / fid,
                    ts=fid, pred_dir=pred_dir,
                    static_agg=static_agg, model=model, device=device,
                    grid_size=0.1, reuse_pred=True,
                    x_range=x_range, y_range=y_range, z_range=z_range,
                    occ_voxel=args.occ_voxel,
                    occ_min_points=args.occ_min_points,
                    export_points=args.export_points,
                    max_export_points=args.max_export_points,
                    ego_filter={"enabled": True,
                                 "x_range": (-3.6, 3.6), "y_range": (-1.2, 1.2),
                                 "min_height": 0.35, "max_height": 4.0,
                                 "ground_fit_margin": 0.5},
                    pose_samples=pose_samples,
                )
                # Cameras meta (same per clip, written into each frame's meta.json)
                meta_path = out_scene / "frames" / fid / "meta.json"
                if meta_path.is_file():
                    last_frame_meta = json.loads(meta_path.read_text())
                # Also stash the raw frame's lidar deskew md5 so store's
                # md5_from_backup() picks it up by first field, without needing
                # backup/<scene_name>/<ts>/frame.json path matching.
                raw_md5 = None
                try:
                    fj = json.loads((clip_dir / "frames" / str(fid) / "frame.json").read_text())
                    sensors = ((fj.get("dependency") or {}).get("sensors") or {})
                    for key in ("lidar_merge_deskew", "lidar_merge", "lidar_merge_nodeskew"):
                        md5 = (sensors.get(key) or {}).get("md5")
                        if md5 and len(md5) == 32:
                            raw_md5 = md5
                            break
                    if not raw_md5 and fj.get("md5") and len(fj["md5"]) == 32:
                        raw_md5 = fj["md5"]
                except Exception:
                    pass
                frame_paths.append({"ts": str(fid),
                                    "path": f"frames/{fid}",
                                    "timestamp": str(fid),
                                    "frame_id": str(fid),
                                    "dir": f"frames/{fid}",
                                    "meta_uri": f"frames/{fid}/meta.json",
                                    **({"raw_md5": raw_md5} if raw_md5 else {})})
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    ERROR frame {fid}: {e}")

        if args.dry_run or not frame_paths:
            status.append({"i": i, "clip_id": clip_id, "rc": 2,
                           "msg": "dry or no frames"})
            continue

        # ---- count n_occ / n_points from binary files for viewer header ----
        for fp in frame_paths:
            fdir = out_scene / fp["dir"]
            occ_bin = fdir / "occ_centers.f32.bin"
            if occ_bin.is_file():
                try: fp["n_occ"] = int(occ_bin.stat().st_size // (4 * 3))
                except Exception: pass
            pts_bin = fdir / "points_xyz.f32.bin"
            if pts_bin.is_file():
                try: fp["n_points"] = int(pts_bin.stat().st_size // (4 * 3))
                except Exception: pass

        # ---- index.json (schema mirrors export_robotruck_occ_scene.main exactly) ----
        coarse = export_mod.coarse_taxonomy()
        cameras_meta = (last_frame_meta or {}).get("cameras", [])
        index_doc = {
            "name": scene_name, "tag": tag, "clip_id": clip_id,
            "frame_count": len(frame_paths),
            "frames": [
                {
                    "frame_id": fp["frame_id"],
                    "timestamp": fp["timestamp"],
                    "meta_uri": fp["meta_uri"],
                    "dir": fp["dir"],
                    "n_occ": int(fp.get("n_occ", 0)),
                    "n_points": int(fp.get("n_points", 0)),
                    **({"raw_md5": fp["raw_md5"]} if fp.get("raw_md5") else {}),
                }
                for fp in frame_paths
            ],
            "cameras": cameras_meta,
            "coarse_taxonomy": coarse,
            "voxel_size": args.occ_voxel,
            "x_range": list(x_range), "y_range": list(y_range), "z_range": list(z_range),
            "class_names": (last_frame_meta or {}).get("class_names", []),
            "class_colors_rgb": (last_frame_meta or {}).get("class_colors_rgb", []),
            "geometry_quality": geometry_quality,
        }
        out_scene.mkdir(parents=True, exist_ok=True)
        (out_scene / "index.json").write_text(json.dumps(index_doc))
        print(f"  wrote index.json -> {out_scene}")

        # ---- store (call store main() via sys.argv override) ----
        rc_store = 0
        if args.write_store:
            try:
                _sys_argv_save = sys.argv[:]
                sys.argv = [
                    "store_robotruck_occ_gridfs.py",
                    "--scene", str(out_scene),
                    "--raw-frame-collection", args.raw_frame_collection,
                    "--raw-clip-collection", args.raw_clip_collection,
                    "--asset-root", str(Path(args.asset_root).resolve()),
                    # backup_root must be ABSOLUTE directory ending in /scene_name's parent
                    # so store can find <backup>/<scene>/frames/<ts>/frame.json
                    "--backup-root", str(Path(args.backup_root).resolve()),
                    "--write",
                ]
                ret = store_mod.main()
                sys.argv[:] = _sys_argv_save
                rc_store = int(ret or 0)
                print(f"  store main() rc={rc_store}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  STORE FAIL: {e}")
                rc_store = 99
                try: sys.argv[:] = _sys_argv_save
                except Exception: pass

        status.append({"i": i, "clip_id": clip_id, "tag": tag,
                       "frames": len(frame_paths), "rc": rc_store})

    # final summary
    print("\n============== SUMMARY ==============")
    for s in status:
        print(f"  [{s.get('i','?'):>2}] rc={s.get('rc','?'):<3} frames={s.get('frames','-'):>4} "
              f"{s.get('clip_id','')}  tag={s.get('tag','')}  {s.get('msg','')}")
    Path("/tmp/random10_inproc_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
