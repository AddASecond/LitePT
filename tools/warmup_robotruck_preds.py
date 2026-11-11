#!/usr/bin/env python3
"""A lane: precompute preds for a materialized clip (agg-stride ∪ export-stride).

Contract: writes {pred_dir}/{ts}_pred.npy (int32, N = lidar points).
See tools/occ/CONTRACTS.md. Production (B) must call this via CLI only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def main(argv: list[str] | None = None, *, model_cache=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--agg-stride", type=int, default=2)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--timestamps", nargs="+", help="Exact export/aggregation timestamps; overrides stride selection")
    ap.add_argument("--config-file", default="configs/waymo/semseg-litept-small-v1m1.py")
    ap.add_argument(
        "--weight",
        default="checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth",
    )
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument(
        "--mode",
        choices=["infer", "fake"],
        default="infer",
        help="fake: label everything class 7 (no CUDA)",
    )
    ap.add_argument("--worker-socket", help="Send this A CLI job to a persistent local model worker")
    ap.add_argument("--seed", type=int, help="Reproducible inference seed, set after model loading")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    if args.worker_socket:
        import socket
        forwarded = argv.copy()
        option = next(i for i, value in enumerate(forwarded)
                      if value == '--worker-socket' or value.startswith('--worker-socket='))
        del forwarded[option:option + (1 if '=' in forwarded[option] else 2)]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1800)
            client.connect(args.worker_socket)
            client.sendall((json.dumps({'argv': forwarded}) + '\n').encode())
            with client.makefile('rb') as stream:
                result = json.loads(stream.readline(4 * 1024 * 1024))
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'Prediction worker failed'))
        print(f"persistent prediction complete elapsed={result['elapsed_seconds']:.3f}s", flush=True)
        return 0

    if args.mode == "infer":
        from hami_cuda import setup_cuda_env
        setup_cuda_env()

    import numpy as np
    import infer_robotruck_mongo_frame as h

    clip = Path(args.clip_dir)
    idx = json.loads((clip / "frames_index.json").read_text())
    tss = [str(e["timestamp"]) for e in idx if e.get("has_lidar")]
    need = list(
        dict.fromkeys(
            tss[:: max(1, args.agg_stride)] + tss[:: max(1, args.stride)][: args.max_frames]
        )
    )
    if args.timestamps is not None:
        unknown = set(args.timestamps) - set(tss)
        if unknown:
            raise ValueError(f"requested timestamps absent from clip: {sorted(unknown)}")
        need = list(dict.fromkeys(args.timestamps))

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for ts in need:
        binp = clip / "frames" / ts / "lidar_merge.bin"
        pp = pred_dir / f"{ts}_pred.npy"
        if not binp.is_file():
            continue
        n = binp.stat().st_size // (4 * len(h.LIDAR_COLS))
        if pp.is_file():
            try:
                if np.load(pp).shape[0] == n:
                    continue
            except Exception:
                pass
        todo.append((ts, n))
    print(f"pred_warmup: need={len(need)} todo={len(todo)}", flush=True)
    if args.mode == "fake":
        for ts, n in todo:
            np.save(pred_dir / f"{ts}_pred.npy", np.full(n, 7, np.int32))
        print(f"  fake labels written: {len(todo)}", flush=True)
        return 0
    if not todo:
        return 0

    config_path = (ROOT / args.config_file).resolve()
    weight_path = (ROOT / args.weight).resolve()
    model_key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (config_path, weight_path))
    if model_cache is not None and model_key in model_cache:
        model = model_cache[model_key]
    else:
        if model_cache is not None:
            model_cache.clear()
        model, _ = h.load_segmentor(config_path, weight_path, "cuda")
        if model_cache is not None:
            model_cache[model_key] = model
    if args.seed is not None:
        np.random.seed(args.seed)
        h.torch.manual_seed(args.seed)
    from concurrent.futures import ThreadPoolExecutor

    def prepare(ts):
        pts = h.load_lidar_bin(clip / "frames" / ts / "lidar_merge.bin", num_cols=len(h.LIDAR_COLS))
        coord = pts[:, :3].astype(np.float32)
        strength = np.tanh(pts[:, 3].reshape(-1, 1) / 255.0).astype(np.float32)
        return h.prepare_infer_frame(coord, strength, args.grid_size)

    # A single producer preserves NumPy sampling order and keeps at most one
    # preprocessed frame ahead of the GPU, independent of clip length.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare, todo[0][0])
        for i, (ts, n) in enumerate(todo):
            data = future.result()
            if i + 1 < len(todo):
                future = pool.submit(prepare, todo[i + 1][0])
            pred = h.infer_prepared(model, data, "cuda")
            target = pred_dir / f"{ts}_pred.npy"
            temporary = target.with_suffix('.partial.npy')
            np.save(temporary, pred.astype(np.int32))
            temporary.replace(target)
            print(f"  [{i+1}/{len(todo)}] {ts} N={n} pred={pred.shape[0]}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
