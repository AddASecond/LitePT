"""Infer LitePT on local Robotruck clip backups (no Mongo).

Expects the layout produced by the earlier backup:
  data/robotruck_clips_backup/<clip_name>/frames/<timestamp>/lidar_merge.bin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402


def _load_mongo_helpers():
    path = ROOT / "tools" / "infer_robotruck_mongo_frame.py"
    spec = importlib.util.spec_from_file_location("infer_robotruck_mongo_frame", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_h = _load_mongo_helpers()
LIDAR_COLS = _h.LIDAR_COLS
infer_frame = _h.infer_frame
load_lidar_bin = _h.load_lidar_bin
load_segmentor = _h.load_segmentor
point_clip_mask = _h.point_clip_mask
render_pred_vis = _h.render_pred_vis

from visualize import save_class_legend  # noqa: E402


def pick_frame(clip_dir: Path, timestamp: str | None) -> tuple[Path, str]:
    frames_dir = clip_dir / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)

    if timestamp:
        fr = frames_dir / str(timestamp)
        lidar = fr / "lidar_merge.bin"
        if not lidar.is_file():
            raise FileNotFoundError(lidar)
        return lidar, str(timestamp)

    index_path = clip_dir / "frames_index.json"
    if index_path.is_file():
        entries = json.loads(index_path.read_text())
        with_lidar = [e for e in entries if e.get("has_lidar")]
        if not with_lidar:
            raise RuntimeError(f"No lidar frames in {index_path}")
        mid = with_lidar[len(with_lidar) // 2]
        ts = str(mid["timestamp"])
        lidar = frames_dir / ts / "lidar_merge.bin"
        if not lidar.is_file():
            raise FileNotFoundError(lidar)
        return lidar, ts

    # fallback: first frame dir with lidar_merge.bin
    for fr in sorted(frames_dir.iterdir()):
        lidar = fr / "lidar_merge.bin"
        if lidar.is_file():
            return lidar, fr.name
    raise RuntimeError(f"No lidar_merge.bin under {frames_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backup-root",
        default="data/robotruck_clips_backup",
        help="Root with stop_/rain_ clip dirs",
    )
    ap.add_argument(
        "--clips",
        nargs="*",
        default=[],
        help="Clip dir names; default = all subdirs with frames/",
    )
    ap.add_argument("--timestamp", default="", help="Optional fixed timestamp for all clips")
    ap.add_argument(
        "--config-file",
        default="configs/waymo/semseg-litept-small-v1m1.py",
    )
    ap.add_argument(
        "--weight",
        default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth",
    )
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument(
        "--pc-range",
        type=float,
        nargs=6,
        default=None,
        help="Optional PointClip x0 y0 z0 x1 y1 z1; default=no clip (full cloud)",
    )
    ap.add_argument("--out-dir", default="exp/robotruck/local_clips_vis")
    ap.add_argument("--max-points-vis", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    backup_root = (ROOT / args.backup_root).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clips:
        clip_dirs = [backup_root / c for c in args.clips]
    else:
        clip_dirs = sorted(
            p for p in backup_root.iterdir() if p.is_dir() and (p / "frames").is_dir()
        )
    if not clip_dirs:
        print(f"No clip dirs under {backup_root}")
        return 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_segmentor(ROOT / args.config_file, ROOT / args.weight, device)

    legend_path = out_dir / "waymo_class_legend.png"
    save_class_legend(legend_path)
    print(f"legend -> {legend_path}")

    for i, clip_dir in enumerate(clip_dirs):
        if not clip_dir.is_dir():
            print(f"skip missing clip: {clip_dir}")
            continue
        lidar_path, ts = pick_frame(clip_dir, args.timestamp or None)
        print(f"\n=== {clip_dir.name} ts={ts} ===")
        print(f"lidar: {lidar_path}")

        pts = load_lidar_bin(lidar_path, num_cols=len(LIDAR_COLS))
        coord = pts[:, :3]
        intensity = pts[:, 3]
        if args.pc_range is not None:
            mask = point_clip_mask(coord, args.pc_range)
            coord = coord[mask]
            intensity = intensity[mask]
            print(f"points after clip: {coord.shape[0]} (from {pts.shape[0]})")
        else:
            print(f"points (full cloud, no clip): {coord.shape[0]}")
        strength = np.tanh(intensity.reshape(-1, 1) / 255.0).astype(np.float32)

        pred = infer_frame(model, coord, strength, device, grid_size=args.grid_size)
        assert pred.shape[0] == coord.shape[0]

        stem = f"{clip_dir.name}_{ts}"
        pred_path = out_dir / f"{stem}_pred.npy"
        np.save(pred_path, pred.astype(np.int32))
        np.save(out_dir / f"{stem}_coord.npy", coord.astype(np.float32))

        vis_path = out_dir / f"{stem}_pred_vis.png"
        render_pred_vis(
            coord,
            pred,
            intensity,
            vis_path,
            title=f"Robotruck local {clip_dir.name} ts={ts}",
            max_points=args.max_points_vis,
            seed=args.seed + i,
        )
        print(f"pred -> {pred_path}")
        print(f"vis  -> {vis_path}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
