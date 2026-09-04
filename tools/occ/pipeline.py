#!/usr/bin/env python3
"""Run raw-volume LitePT inference, then index content-addressed OCC in MongoDB."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def _setup_cuda_env() -> None:
    # Must run BEFORE any `subprocess.run(python ...)` that uses CUDA, and
    # before any import of torch (materialize stage is CPU, but we set it here
    # so children inherit fixed env for the export stage).
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
    # Warm up CUDA in THIS process too: when we later exec the exporter via
    # subprocess (fork+exec), HAMI's libvgpu session state for the cgroup has
    # already been primed by our own cuInit call, which eliminates one of the
    # main 304-producing race windows.  Ignore result – subprocess is what
    # actually needs it.
    if os.environ.get("LITEPT_SKIP_CUDA_WARMUP") != "1":
        try:
            import torch as _torch
            _ = _torch.cuda.is_available()
        except Exception:
            pass


_setup_cuda_env()


import numpy as np
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("occ_gridfs", ROOT / "tools/occ/_impl/store.py")
STORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STORE)
INGEST = STORE.INGEST


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str, ensure_ascii=False, indent=2))


def pcd_binary_to_litept(source: Path, target: Path) -> None:
    """Convert binary PCD x/y/z/intensity/ring/timestamp/lidar_id to float32 Nx7."""
    with source.open("rb") as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD has no DATA line: {source}")
            text = line.decode("ascii", errors="strict").strip()
            if text and not text.startswith("#"):
                key, *values = text.split()
                header[key.upper()] = values
            if text.upper().startswith("DATA "):
                break
        if header.get("DATA") != ["binary"]:
            raise ValueError(f"only DATA binary PCD is supported: {source}")
        fields = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        if any(count != 1 for count in counts):
            raise ValueError(f"PCD COUNT != 1 is unsupported: {source}")
        mapping = {
            ("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1",
            ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
            ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
        }
        dtype = np.dtype([(name, mapping[(kind, size)]) for name, kind, size in zip(fields, types, sizes)])
        points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
        raw = stream.read()
    expected = points * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"PCD byte size mismatch: {source}: {len(raw)} != {expected}")
    data = np.frombuffer(raw, dtype=dtype, count=points)
    required = ("x", "y", "z")
    if any(name not in data.dtype.names for name in required):
        raise ValueError(f"PCD lacks xyz: {source}")
    output = np.zeros((points, 7), dtype=np.float32)
    output[:, 0] = data["x"]
    output[:, 1] = data["y"]
    output[:, 2] = data["z"]
    if "intensity" in data.dtype.names:
        output[:, 3] = data["intensity"]
    if "ring" in data.dtype.names:
        output[:, 4] = data["ring"]
    if "lidar_id" in data.dtype.names:
        output[:, 6] = data["lidar_id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(target)


def link_or_convert_lidar(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".bin":
        target.symlink_to(source)
    elif source.suffix == ".pcd":
        pcd_binary_to_litept(source, target)
    else:
        raise ValueError(source)


def materialize_raw_cache(db, frame_collection: str, clip_collection: str, clip_id: str, cache: Path) -> int:
    raw_clip = db[clip_collection].find_one({"clip_id": clip_id})
    if raw_clip is None:
        raise RuntimeError(f"clip_id={clip_id} absent from {clip_collection}")
    frames = list(db[frame_collection].find({"clip_id": clip_id}).sort("timestamp", 1))
    if not frames:
        raise RuntimeError(f"no frames for clip_id={clip_id} in {frame_collection}")
    index = []
    for number, raw in enumerate(frames, 1):
        ts = str(raw["timestamp"])
        md5 = INGEST.raw_lidar_md5(raw) or raw.get("md5")
        lidar = STORE.resolve_raw(md5, "lidar")
        frame_dir = cache / "frames" / ts
        link_or_convert_lidar(lidar, frame_dir / "lidar_merge.bin")
        sensors = ((raw.get("dependency") or {}).get("sensors") or {})
        copied_cameras = {}
        for name, sensor in sensors.items():
            if not name.startswith("camera") or not isinstance(sensor, dict):
                continue
            cam_md5 = sensor.get("md5")
            if not isinstance(cam_md5, str) or not INGEST.MD5_RE.fullmatch(cam_md5):
                continue
            camera = STORE.resolve_raw(cam_md5, "camera")
            target = frame_dir / f"{name}.jpg"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(camera)
            copied_cameras[name] = target.name
        json_write(frame_dir / "frame.json", raw)
        index.append({"timestamp": raw["timestamp"], "md5": md5, "has_lidar": True, "n_cameras": len(copied_cameras), "copied": {"lidar_merge": "lidar_merge.bin", "cameras": copied_cameras}})
        if number % 50 == 0 or number == len(frames):
            print(f"materialized raw {number}/{len(frames)}", flush=True)
    json_write(cache / "clip.json", raw_clip)
    json_write(cache / "frames_index.json", index)
    json_write(cache / "SUMMARY.json", {"clip_id": clip_id, "mongo_frame_count": len(frames), "raw_roots": [str(p) for p in STORE.RAW_ROOTS]})
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-frame-collection", required=True)
    ap.add_argument("--raw-clip-collection", default="")
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--scene-name", default="")
    ap.add_argument("--database", default="perception_experiment")
    ap.add_argument("--mongo-uri", default=INGEST.DEFAULT_URI)
    ap.add_argument("--cache-root", default="exp/robotruck/raw_volume_cache")
    ap.add_argument("--scene-root", default="exp/robotruck/occ_scenes")
    ap.add_argument("--asset-root", default="/data/rawdata-4/occupancy")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--force-infer", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    raw_clip_collection = args.raw_clip_collection or INGEST.infer_collection(args.raw_frame_collection, "frames", "clips")
    scene_name = args.scene_name or f"mongo_{args.raw_frame_collection}_{args.clip_id}"
    cache = (ROOT / args.cache_root / scene_name).resolve()
    scene = (ROOT / args.scene_root / scene_name).resolve()
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]
    if args.force_infer or not (scene / "index.json").is_file():
        materialize_raw_cache(db, args.raw_frame_collection, raw_clip_collection, args.clip_id, cache)
        python = ROOT / ".venv_smoke/bin/python"
        command = [str(python), str(ROOT / "tools/occ/_impl/export_scene.py"), "--clip", scene_name, "--backup-root", args.cache_root, "--out-dir", args.scene_root, "--stride", str(args.stride), "--max-frames", str(args.max_frames), "--export-points", "--aggregate-static"]
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        print(f"reuse existing inference scene: {scene}", flush=True)
    store_command = [str(ROOT / ".venv_smoke/bin/python"), str(ROOT / "tools/occ/_impl/store.py"), "--scene", str(scene), "--raw-frame-collection", args.raw_frame_collection, "--raw-clip-collection", raw_clip_collection, "--backup-root", args.cache_root, "--asset-root", args.asset_root]
    if args.write:
        store_command.append("--write")
    subprocess.run(store_command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
