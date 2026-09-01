#!/usr/bin/env python3
"""Stage 2 — FAST video generator.

GOAL: produce a playable .mp4 per clip using the existing K / T / poses /
projection pipeline.  We DO NOT tweak intrinsics, distortion, or any
calibration parameter.  If projection is off, it shows in the video — that's
for Stage 3 to document, not here to fix.

Strategy (fast path, not robust):
  * Iterate the manifest produced by Stage 1 (cone_filter_denoise.py), or
    iterate a --clips-json list, or if none, process all materialized clips.
  * For each clip, call tools/render_robotruck_clip_video.py.main() via
    sys.argv override (same pattern used elsewhere in the repo).
  * All env / projection defaults are accepted as-is.  If some cameras are
    missing / some frames have no cones, video still gets produced.
  * If render fails on a clip, the error is logged and we continue.

Usage:
    ./.venv_smoke/bin/python tools/cone_render_fast.py \
        --manifest exp/robotruck/cone_mid/manifest.json \
        --backup-root exp/robotruck/raw_volume_cache \
        --out-dir exp/robotruck/cone_videos
"""
from __future__ import annotations
import argparse, json, logging, os, subprocess, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cone_render_fast")


def _cuda_env():
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.pop("_CUDA_COMPAT_PATH", None)
        os.environ.pop("Path", None)
    except Exception: pass


_cuda_env()


# --------- HARD-CODED FAST RENDER TUNES (DO NOT TUNE) -----------------------
# Keep them modest so we finish on-time.  Quality / frame rate correctness
# explicitly de-prioritized vs. producing a file.
STRIDE = 3                # render 1 frame every 3 -> ~100 frames per clip
MAX_FRAMES = 120          # hard cap
FPS = 10.0
# 5 cams side-by-side -> final frame is 5x wide; keep total <= 4K (3840x2160)
# because OpenCV FFMPEG VideoWriter fails to init on 9600x5568 frames.
TILE_W = 768
TILE_H = 432
PROJ_RADIUS = 4
AGG_STRIDE = 4
STATIC_VOXEL = 0.4
OCC_VOXEL = 0.4
# leave camera intrinsic alone
DEVICE = "cuda"
# ---------------------------------------------------------------------------

def run_render(clip: str, backup_root: Path, out_dir: Path,
               config_file: Path, weight: Path) -> tuple[int, str]:
    """Run render_robotruck_clip_video.main() by sys.argv override.

    Returns (rc, out_path_or_error).
    """
    import importlib.util
    path = ROOT / "tools" / "render_robotruck_clip_video.py"
    spec = importlib.util.spec_from_file_location(
        "render_robotruck_clip_video_fast", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(mod)  # _setup_cuda_env etc runs as side-effect
    except SystemExit:
        pass
    except Exception as exc:
        return 98, f"import render module failed: {exc}"

    out_dir.mkdir(parents=True, exist_ok=True)
    # pick a recognizable filename
    out_name = f"cone_fast__{clip[:8]}"

    _save = sys.argv[:]
    sys.argv = [
        "render_robotruck_clip_video.py",
        "--clip", clip,
        "--backup-root", str(backup_root.resolve()),
        "--out-dir", str(out_dir.resolve()),
        "--config-file", str(config_file.resolve()),
        "--weight", str(weight.resolve()),
        "--device", DEVICE,
        "--stride", str(STRIDE),
        "--max-frames", str(MAX_FRAMES),
        "--fps", str(FPS),
        "--tile-w", str(TILE_W),
        "--tile-h", str(TILE_H),
        "--proj-radius", str(PROJ_RADIUS),
        "--agg-stride", str(AGG_STRIDE),
        "--static-voxel", str(STATIC_VOXEL),
        "--occ-voxel", str(OCC_VOXEL),
        "--reuse-pred",
        "--out-name", out_name,
        # keep panels but skip costly extras
        "--no-point-viz",
        "--no-cam-occ-side",
    ]
    try:
        rc = int(mod.main() or 0)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 99
    except Exception as exc:
        rc = 99
        log.error("render %s threw: %s\n%s", clip[:8], exc, traceback.format_exc())
    finally:
        sys.argv[:] = _save
    # try to locate the produced video (render writes into out_dir/<clip>/)
    candidate = None
    for p in sorted((out_dir).rglob(f"*{out_name}*.mp4"), reverse=True):
        candidate = str(p); break
    return rc, (candidate or f"<no mp4 found rc={rc}>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("exp/robotruck/cone_mid/manifest.json"))
    ap.add_argument("--clips-json", type=Path,
                    help="Fall back if manifest missing.  JSON {clips:[uuid, ...]}")
    ap.add_argument("--backup-root", type=Path,
                    default=Path("exp/robotruck/raw_volume_cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("exp/robotruck/cone_videos"))
    ap.add_argument("--config-file", type=Path,
                    default=Path("configs/waymo/semseg-litept-small-v1m1.py"))
    ap.add_argument("--weight", type=Path,
                    default=Path("checkpoints/waymo-semseg-litept-small-v1m1/model/model_best.pth"))
    ap.add_argument("--max-clips", type=int, default=999)
    args = ap.parse_args()

    clips = []
    if args.manifest.is_file():
        try:
            m = json.loads(args.manifest.read_text())
            clips = [row["clip_id"] for row in m.get("clips", [])]
        except Exception as exc:
            log.warning("manifest parse failed: %s", exc)
    if not clips and args.clips_json and args.clips_json.is_file():
        try:
            clips = list((json.loads(args.clips_json.read_text())
                        .get("clips") or []))
        except Exception as exc:
            log.warning("clips-json parse failed: %s", exc)
    if not clips:
        log.error("No clips to render (manifest + clips-json both empty).  Exit.")
        return 2
    clips = clips[:args.max_clips]

    log.info("render queue size=%d backup=%s out=%s", len(clips), args.backup_root, args.out_dir)
    results = []
    ok = fail = 0
    for i, c in enumerate(clips, 1):
        log.info("[%d/%d] render %s", i, len(clips), c[:8])
        rc, info = run_render(c, args.backup_root, args.out_dir,
                              args.config_file, args.weight)
        results.append({"clip_id": c, "render_rc": rc, "info": str(info)})
        if rc == 0 and str(info).endswith(".mp4") and Path(str(info)).is_file():
            ok += 1
            log.info("  OK -> %s", Path(str(info)).name)
        else:
            fail += 1
            log.warning("  FAIL rc=%d info=%s", rc, info)

    out_json = args.out_dir / "render_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"n_requested": len(clips), "ok": ok, "fail": fail,
         "results": results}, indent=2))
    log.info("DONE  ok=%d  fail=%d  summary=%s", ok, fail, out_json)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    try: sys.exit(main())
    except KeyboardInterrupt: sys.exit(130)
