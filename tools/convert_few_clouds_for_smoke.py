#!/usr/bin/env python3
"""Convert 1–few point clouds into a tiny LitePT Waymo-style folder for smoke tests.

Does NOT copy the full dataset. Writes only:
  data/waymo_smoke/training/<name>/<id>/{coord,strength,pose,segment}.npy

Sources:
  --from-hf-demo     HuggingFace prs-eth/LitePT_demo outdoor bin (default)
  --from-bin PATH    local .bin (float32); tries cols 4/5/7/8
  --max-points N     optional random subsample for 18GB safety

segment.npy is filled with -1 (ignore). Enough to verify segmentation inference;
real Waymo labels come later from preprocess_waymo.py / TFRecords.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def load_bin(path: str, cols: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.fromfile(path, dtype=np.float32)
    if cols is not None:
        if arr.size % cols != 0:
            raise ValueError(f"{path}: size {arr.size} not divisible by cols={cols}")
        pts = arr.reshape(-1, cols)
    else:
        pts = None
        for c in (5, 4, 7, 8, 6):
            if arr.size % c == 0 and arr.size // c > 100:
                cand = arr.reshape(-1, c)
                # heuristic: xyz should look metric-ish
                if np.isfinite(cand[:, :3]).all():
                    pts = cand
                    cols = c
                    break
        if pts is None:
            raise ValueError(f"{path}: cannot infer point columns (size={arr.size})")
    coord = pts[:, :3].astype(np.float32)
    if pts.shape[1] >= 4:
        strength = pts[:, 3:4].astype(np.float32)
        # LitePT Waymo preprocess uses tanh(intensity); HF demo scales /255.
        # Keep raw-ish but squash outliers for outdoor bins.
        if float(np.nanmax(np.abs(strength))) > 2.0:
            strength = np.tanh(strength)
        else:
            strength = np.clip(strength, 0.0, 1.0)
    else:
        strength = np.zeros((coord.shape[0], 1), dtype=np.float32)
    return coord, strength


def load_hf_demo() -> tuple[np.ndarray, np.ndarray, str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="prs-eth/LitePT_demo",
        filename="outdoor_sample1.bin",
        repo_type="dataset",
        revision="main",
    )
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    coord = points[:, :3].astype(np.float32)
    strength = (points[:, 3:4] / 255.0).astype(np.float32)
    return coord, strength, "hf_outdoor_sample1"


def write_sample(
    out_root: Path,
    split: str,
    seq: str,
    frame_id: str,
    coord: np.ndarray,
    strength: np.ndarray,
    pose: np.ndarray | None = None,
) -> Path:
    dst = out_root / split / seq / frame_id
    dst.mkdir(parents=True, exist_ok=True)
    if pose is None:
        pose = np.eye(4, dtype=np.float32)
    segment = np.full((coord.shape[0],), -1, dtype=np.int32)
    np.save(dst / "coord.npy", coord)
    np.save(dst / "strength.npy", strength)
    np.save(dst / "pose.npy", pose.astype(np.float32))
    np.save(dst / "segment.npy", segment)
    return dst


def maybe_subsample(
    coord: np.ndarray, strength: np.ndarray, max_points: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or coord.shape[0] <= max_points:
        return coord, strength
    rng = np.random.default_rng(seed)
    idx = rng.choice(coord.shape[0], size=max_points, replace=False)
    return coord[idx], strength[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        default=str(Path(__file__).resolve().parents[1] / "data" / "waymo_smoke"),
    )
    ap.add_argument("--from-hf-demo", action="store_true", default=True)
    ap.add_argument("--no-hf-demo", action="store_true")
    ap.add_argument(
        "--from-bin",
        action="append",
        default=[],
        help="Local .bin path(s); can pass multiple times (max a few).",
    )
    ap.add_argument("--bin-cols", type=int, default=None)
    ap.add_argument("--max-points", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    written = []

    use_hf = args.from_hf_demo and not args.no_hf_demo and not args.from_bin
    if use_hf or (args.from_hf_demo and not args.no_hf_demo and not args.from_bin):
        # default path: HF demo only
        pass

    samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    if args.from_bin:
        for i, p in enumerate(args.from_bin[:5]):
            coord, strength = load_bin(p, cols=args.bin_cols)
            name = Path(p).stem[:24]
            samples.append((coord, strength, f"local_{i}_{name}"))
    elif not args.no_hf_demo:
        coord, strength, name = load_hf_demo()
        samples.append((coord, strength, name))
    else:
        raise SystemExit("Provide --from-bin or allow HF demo (default).")

    for i, (coord, strength, name) in enumerate(samples):
        coord, strength = maybe_subsample(coord, strength, args.max_points, args.seed + i)
        dst = write_sample(
            out_root,
            split="training",
            seq=name,
            frame_id="000000",
            coord=coord,
            strength=strength,
        )
        written.append((dst, coord.shape[0]))
        print(f"wrote {dst}  N={coord.shape[0]}")

    print(f"done: {len(written)} sample(s) under {out_root}")
    print("Next: python tools/smoke_seg_infer.py --data-root", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
