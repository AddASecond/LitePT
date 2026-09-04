"""LitePT repo paths + B helpers (lidar IO, A-CLI pred ensure). No import of infer_*."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

OCC = Path(__file__).resolve().parent
ROOT = OCC.parents[1]

RAW_ROOTS = tuple(Path(f"/data/rawdata{s}") for s in ("", "-1", "-2", "-3", "-4"))
DEFAULT_ASSET_ROOT = Path("/data/rawdata-4/occupancy")
DEFAULT_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)
LIDAR_COLS = ("x", "y", "z", "intensity", "ring", "dt", "lidar_id")


def ensure_import_path(*, tools: bool = False, repo: bool = False) -> Path:
    for path in (OCC, *((ROOT / "tools",) if tools else ()), *((ROOT,) if repo else ())):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return ROOT


def setup_cuda_env(*, warmup: bool = True) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("_CUDA_COMPAT_PATH", None)
    os.environ.pop("Path", None)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/usr/lib/x86_64-linux-gnu" not in ld.split(":"):
        head = "/usr/lib/x86_64-linux-gnu"
        cudalib = "/usr/local/cuda/targets/x86_64-linux/lib"
        os.environ["LD_LIBRARY_PATH"] = f"{head}:{cudalib}" + (f":{ld}" if ld else "")
    os.environ["HAMI_DISABLE_WARN"] = "1"
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0+PTX"
    if not warmup or os.environ.get("LITEPT_SKIP_CUDA_WARMUP") == "1":
        return
    try:
        import torch
        torch.cuda.is_available()
    except Exception:
        pass


def load_lidar_bin(path: Path, num_cols: int = len(LIDAR_COLS)) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % num_cols != 0:
        raise ValueError(f"{path}: float32 count {arr.size} not divisible by {num_cols}")
    return arr.reshape(-1, num_cols)


def ensure_preds(
    *,
    clip_dir: Path,
    pred_dir: Path,
    timestamps: list[str],
    agg_timestamps: list[str] | None = None,
    stride: int = 2,
    agg_stride: int = 5,
    max_frames: int = 0,
    config_file: str = "configs/waymo/semseg-litept-small-v1m1.py",
    weight: str = "checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth",
    grid_size: float = 0.05,
) -> None:
    """Fill missing preds via A CLI only (no import of infer modules)."""
    need = list(dict.fromkeys(list(timestamps) + list(agg_timestamps or [])))
    missing = []
    for ts in need:
        pp = pred_dir / f"{ts}_pred.npy"
        binp = clip_dir / "frames" / ts / "lidar_merge.bin"
        if not binp.is_file():
            continue
        n = binp.stat().st_size // (4 * len(LIDAR_COLS))
        ok = False
        if pp.is_file():
            try:
                ok = int(np.load(pp).shape[0]) == n
            except Exception:
                ok = False
        if not ok:
            missing.append(ts)
    if not missing:
        return
    script = ROOT / "tools" / "warmup_robotruck_preds.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing preds ({len(missing)}) and A CLI {script}")
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--clip-dir", str(clip_dir), "--pred-dir", str(pred_dir),
        "--stride", str(max(1, stride)), "--agg-stride", str(max(1, agg_stride)),
        "--max-frames", str(max_frames if max_frames > 0 else max(len(timestamps), 1)),
        "--config-file", config_file, "--weight", weight,
        "--grid-size", str(grid_size), "--mode", "infer",
    ]
    print(f"[ensure_preds] missing={len(missing)} -> A CLI", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"A warmup CLI failed rc={proc.returncode}")
