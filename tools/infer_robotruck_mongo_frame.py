"""Infer LitePT (Waymo-S) on one Robotruck frame from Mongo + local lidar bin, then visualize.

Data source:
  Mongo  perception_experiment.raw_data_clips_lidar14_0731  (clip meta)
         perception_experiment.raw_data_frames_lidar14_0731 (frame + lidar_merge.md5)
  Disk   /data/rawdata/lidar/{md5[0:2]}/{md5[2:4]}/{md5[4:]}.bin
         float32 Nx7: x,y,z,intensity,ring,dt,lidar_id

Robotruck frames typically have no dense semseg GT, so the PNG is Pred (BEV/side)
plus intensity-colored reference views. Class colors match visualize.py Waymo palette.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path


def _setup_cuda_env() -> None:
    """HAMI-vGPU container hardening: fix env BEFORE torch/spconv import.

    Canonical rules (mirrors LitePT/.cuda_env.sh):
      - nvidia0 node missing, only nvidia1 exists (CUDA enumerates it as 0):
        CUDA_VISIBLE_DEVICES=0
      - _CUDA_COMPAT_PATH forces old compat (575) against newer driver (580) -> remove
      - Windows "Path" (capital P) leak must go
      - prefer driver libcuda dirs in LD_LIBRARY_PATH
      - FORCE EARLY CUDA INIT via torch before cv2/matplotlib or any other
        CUDA-linked library can run cuInit first.  cv2 opencv-python-cuda
        builds make hidden cuDeviceGetCount calls on import and, under HAMI
        libvgpu.so, those are known to return error 304 intermittently and
        then poison torch's subsequent cudaGetDeviceCount() call.  Grabbing
        the CUDA session via torch first is the known-good pattern.
    This is in plain Python (no shell) so it works via `python file.py`.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("_CUDA_COMPAT_PATH", None)
    os.environ.pop("Path", None)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/usr/lib/x86_64-linux-gnu" not in ld.split(":"):
        head = "/usr/lib/x86_64-linux-gnu"
        cudalib = "/usr/local/cuda/targets/x86_64-linux/lib"
        if ld:
            os.environ["LD_LIBRARY_PATH"] = f"{head}:{cudalib}:{ld}"
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{head}:{cudalib}"
    # HAMI subprocess vgpu-alloc stability hints
    os.environ["HAMI_DISABLE_WARN"] = "1"
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0+PTX"

    # -- Immediate early init -- we import torch here precisely to be the
    # first CUDA caller in the process, before cv2/matplotlib hit their own
    # hidden CUDA probe calls (which would race against us on HAMI vGPU).
    import torch as _torch
    try:
        if not _torch.cuda.is_available():
            # Even if unavailable, record this cleanly so downstream failures
            # are "no CUDA" rather than the misleading cudaGetDeviceCount 304.
            pass
    except Exception:
        pass


_setup_cuda_env()


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.transform import Compose  # noqa: E402
from models.builder import build_model  # noqa: E402
from utils.config import Config  # noqa: E402
from visualize import WAYMO_COLORS, WAYMO_NAMES, labels_to_colors, save_class_legend  # noqa: E402

# Prefer env; do not hardcode credentials in the repo.
DEFAULT_URI = os.environ.get(
    "ROBOTRUCK_MONGO_URI",
    "mongodb://krk030-mongodb:27017/?authSource=perception_experiment",
)
LIDAR_COLS = ("x", "y", "z", "intensity", "ring", "dt", "lidar_id")


def md5_to_lidar_path(md5: str, rawdata_root: Path) -> Path:
    return rawdata_root / "lidar" / md5[:2] / md5[2:4] / f"{md5[4:]}.bin"


def load_lidar_bin(path: Path, num_cols: int = 7) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % num_cols != 0:
        raise ValueError(f"{path}: float32 count {arr.size} not divisible by {num_cols}")
    return arr.reshape(-1, num_cols)


def find_frame_with_local_bin(
    uri: str,
    clip_coll: str,
    frame_coll: str,
    rawdata_root: Path,
    clip_id: str | None,
    max_scan: int,
):
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    # URI uses authSource=perception_experiment but may omit /dbname — pin explicitly.
    db = client["perception_experiment"]
    clips = db[clip_coll]
    frames = db[frame_coll]

    clip_query = {"clip_id": clip_id} if clip_id else {}
    # Prefer scanning local bins then $in query (fast); fall back to clip→frames.
    local_md5s = []
    lidar_root = rawdata_root / "lidar"
    for p in lidar_root.glob("*/*/*.bin"):
        local_md5s.append(p.parts[-3] + p.parts[-2] + p.stem)
        if len(local_md5s) >= max_scan:
            break
    if not local_md5s:
        raise FileNotFoundError(f"No lidar bins under {lidar_root}")

    frame_q = {"dependency.sensors.lidar_merge.md5": {"$in": local_md5s}}
    if clip_id:
        frame_q["clip_id"] = clip_id
    proj = {
        "md5": 1,
        "timestamp": 1,
        "clip_id": 1,
        "bag_name": 1,
        "tag": 1,
        "dependency.sensors.lidar_merge.md5": 1,
        "dependency.ego_pose": 1,
        "groundtruth": 1,
    }
    frame = frames.find_one(frame_q, proj)
    if frame is None and clip_id:
        raise RuntimeError(f"No local lidar bin for clip_id={clip_id}")
    if frame is None:
        # try any frame from this clip collection's clip_ids that has local data
        for clip in clips.find(clip_query, {"clip_id": 1, "clip_name": 1}).limit(50):
            frame = frames.find_one(
                {
                    "clip_id": clip["clip_id"],
                    "dependency.sensors.lidar_merge.md5": {"$in": local_md5s},
                },
                proj,
            )
            if frame is not None:
                break
    if frame is None:
        raise RuntimeError(
            f"No overlapping local lidar bin for {frame_coll} "
            f"(scanned {len(local_md5s)} local md5s)."
        )

    md5 = frame["dependency"]["sensors"]["lidar_merge"]["md5"]
    path = md5_to_lidar_path(md5, rawdata_root)
    if not path.is_file():
        raise FileNotFoundError(path)

    clip = clips.find_one({"clip_id": frame["clip_id"]}) or {}
    meta = {
        "clip_id": frame.get("clip_id"),
        "clip_name": clip.get("clip_name"),
        "timestamp": frame.get("timestamp"),
        "bag_name": frame.get("bag_name"),
        "tag": frame.get("tag"),
        "md5": md5,
        "lidar_path": str(path),
        "has_semseg_gt": bool(
            isinstance(frame.get("groundtruth"), dict)
            and frame["groundtruth"].get("lidar_semseg")
        ),
    }
    return frame, path, meta


def point_clip_mask(coord: np.ndarray, pc_range) -> np.ndarray:
    x0, y0, z0, x1, y1, z1 = pc_range
    return (
        (coord[:, 0] >= x0)
        & (coord[:, 0] <= x1)
        & (coord[:, 1] >= y0)
        & (coord[:, 1] <= y1)
        & (coord[:, 2] >= z0)
        & (coord[:, 2] <= z1)
    )


def load_segmentor(config_file: Path, weight: Path, device: torch.device):
    cfg = Config.fromfile(str(config_file))
    model = build_model(cfg.model)
    ckpt = torch.load(str(weight), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v
    info = model.load_state_dict(cleaned, strict=True)
    print(f"loaded weight: {weight}  missing={info.missing_keys} unexpected={info.unexpected_keys}")
    model.to(device)
    model.eval()
    return model, cfg


def build_infer_transform(grid_size: float = 0.05) -> Compose:
    # Do not pass "segment" — DefaultSegmentorV2 would enter loss path.
    return Compose(
        [
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "inverse"),
                feat_keys=("coord", "strength"),
            ),
        ]
    )


@torch.no_grad()
def infer_frame(
    model,
    coord: np.ndarray,
    strength: np.ndarray,
    device: torch.device,
    grid_size: float,
) -> np.ndarray:
    data = dict(coord=coord.astype(np.float32), strength=strength.astype(np.float32))
    transform = build_infer_transform(grid_size)
    data = transform(data)
    for k, v in list(data.items()):
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device, non_blocking=True)
    if "offset" not in data:
        data["offset"] = torch.tensor([data["coord"].shape[0]], device=device, dtype=torch.long)
    out = model(data)
    logits = out["seg_logits"]  # [N_down, C]
    pred_down = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    inverse = data["inverse"].cpu().numpy()
    return pred_down[inverse]


def intensity_colors(intensity: np.ndarray) -> np.ndarray:
    x = np.clip(intensity.astype(np.float32) / 255.0, 0.0, 1.0)
    # dark→cyan heatmap for reference panel
    c = np.stack([0.1 + 0.2 * x, 0.3 + 0.6 * x, 0.4 + 0.6 * x], axis=-1)
    return c


def draw_roi_guides(
    ax,
    plane: str,
    *,
    lateral_m: float = 24.0,
    range_marks_m: tuple[float, ...] = (20.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0),
    behind_marks_m: tuple[float, ...] = (25.0, 50.0, 100.0, 150.0, 200.0, 250.0),
    every_m: float = 50.0,
    max_range_m: float = 400.0,
    behind_m: float = 250.0,
) -> None:
    """Overlay ROI on plot axes where +plot_x = vehicle forward (+y).

    Robotruck / imu frame: +y forward, +x lateral. Visualization remaps so
    forward is the plot horizontal axis for readability.
    """
    from matplotlib.patches import Rectangle

    # every-50m grid along forward (plot x), both ahead and behind
    for d in np.arange(every_m, max_range_m + 1e-6, every_m):
        ax.axvline(float(d), color="#888888", lw=0.8, alpha=0.55, zorder=3)
    for d in np.arange(every_m, behind_m + 1e-6, every_m):
        ax.axvline(float(-d), color="#555555", lw=0.7, alpha=0.45, zorder=3)

    # highlighted forward marks
    for d in range_marks_m:
        ax.axvline(float(d), color="#ffcc66", lw=1.2, alpha=0.95, zorder=4)
        ax.text(
            float(d),
            0.98,
            f"{int(d)}m",
            color="#ffcc66",
            fontsize=7,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            zorder=5,
        )

    # highlighted behind marks (negative forward)
    for d in behind_marks_m:
        if d > behind_m + 1e-6:
            continue
        ax.axvline(float(-d), color="#99ddff", lw=1.2, alpha=0.95, zorder=4)
        ax.text(
            float(-d),
            0.98,
            f"-{int(d)}m",
            color="#99ddff",
            fontsize=7,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            zorder=5,
        )

    if plane == "bev":
        # lateral ±24m as horizontal lines on plot (vehicle ±x)
        ax.axhline(lateral_m, color="#66ccff", lw=1.2, alpha=0.95, zorder=4)
        ax.axhline(-lateral_m, color="#66ccff", lw=1.2, alpha=0.95, zorder=4)
        ax.add_patch(
            Rectangle(
                (0.0, -lateral_m),
                max_range_m,
                2.0 * lateral_m,
                fill=False,
                edgecolor="#66ccff",
                linewidth=1.4,
                linestyle="--",
                zorder=4,
            )
        )
        ax.text(
            2.0,
            lateral_m,
            f"±{int(lateral_m)}m",
            color="#66ccff",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=5,
        )


def render_pred_vis(
    coord: np.ndarray,
    pred: np.ndarray,
    intensity: np.ndarray,
    out_png: Path,
    title: str,
    max_points: int,
    seed: int,
    *,
    lateral_m: float = 24.0,
    range_marks_m: tuple[float, ...] = (20.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0),
    behind_marks_m: tuple[float, ...] = (25.0, 50.0, 100.0, 150.0, 200.0, 250.0),
    every_m: float = 50.0,
    max_range_m: float = 400.0,
    behind_m: float = 250.0,
) -> None:
    """Render Pred/Intensity BEV+side with Robotruck axes (+y forward, +x lateral)."""
    n = coord.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        coord = coord[idx]
        pred = pred[idx]
        intensity = intensity[idx]

    pred_c = labels_to_colors(pred)
    int_c = intensity_colors(intensity)

    # Remap for display: plot_x = vehicle y (forward), plot_y_bev = vehicle x (lateral)
    fwd = coord[:, 1]
    lat = coord[:, 0]
    up = coord[:, 2]

    # Cover full cloud (+ at least ROI forward/behind)
    pad = 10.0
    x0 = float(min(fwd.min(), -behind_m) - pad)
    x1 = float(max(fwd.max(), max_range_m) + pad)
    y0 = float(lat.min() - pad)
    y1 = float(lat.max() + pad)
    z0 = float(min(up.min(), -5.0) - 2.0)
    z1 = float(max(up.max(), 12.0) + 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(22, 12), dpi=130)
    panels = [
        (axes[0, 0], fwd, lat, pred_c, "LitePT Pred · BEV", "y forward (m)", "x lateral (m)", "bev"),
        (axes[0, 1], fwd, lat, int_c, "Intensity · BEV", "y forward (m)", "x lateral (m)", "bev"),
        (axes[1, 0], fwd, up, pred_c, "LitePT Pred · Side", "y forward (m)", "z up (m)", "side"),
        (axes[1, 1], fwd, up, int_c, "Intensity · Side", "y forward (m)", "z up (m)", "side"),
    ]
    for ax, xs, ys, colors, ttl, xlab, ylab, plane in panels:
        ax.scatter(xs, ys, c=colors, s=0.1, linewidths=0)
        draw_roi_guides(
            ax,
            plane,
            lateral_m=lateral_m,
            range_marks_m=range_marks_m,
            behind_marks_m=behind_marks_m,
            every_m=every_m,
            max_range_m=max_range_m,
            behind_m=behind_m,
        )
        ax.set_xlim(x0, x1)
        if plane == "bev":
            ax.set_ylim(y0, y1)
        else:
            ax.set_ylim(z0, z1)
        ax.set_title(ttl, color="white")
        ax.set_xlabel(xlab, color="white")
        ax.set_ylabel(ylab, color="white")
        ax.set_facecolor("black")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444444")

    # class histogram strip
    uniq, cnt = np.unique(pred, return_counts=True)
    top = sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda t: -t[1])[:8]
    hist = ", ".join(
        f"{WAYMO_NAMES[i] if 0 <= i < len(WAYMO_NAMES) else i}:{c}" for i, c in top
    )

    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"{title}\nN={n}  ROI ±{int(lateral_m)}m(x) / "
        f"forward 0..{int(max_range_m)}m(+y) behind marks "
        f"{'/'.join(str(int(x)) for x in behind_marks_m)}m  "
        f"| every {int(every_m)}m  |  top: {hist}",
        color="white",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mongo-uri", default=DEFAULT_URI)
    ap.add_argument("--clip-collection", default="raw_data_clips_lidar14_0731")
    ap.add_argument("--frame-collection", default="raw_data_frames_lidar14_0731")
    ap.add_argument("--rawdata-root", default="/data/rawdata")
    ap.add_argument("--clip-id", default="", help="Optional clip_id; else auto-pick local bin")
    ap.add_argument("--max-scan-local", type=int, default=3000)
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
        default=(-75.2, -75.2, -4.0, 75.2, 75.2, 4.0),
        help="PointClip range x0 y0 z0 x1 y1 z1 (Waymo-style)",
    )
    ap.add_argument("--out-dir", default="exp/robotruck/lidar14_0731_one_frame/vis")
    ap.add_argument("--max-points-vis", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rawdata_root = Path(args.rawdata_root)
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, lidar_path, meta = find_frame_with_local_bin(
        uri=args.mongo_uri,
        clip_coll=args.clip_collection,
        frame_coll=args.frame_collection,
        rawdata_root=rawdata_root,
        clip_id=args.clip_id or None,
        max_scan=args.max_scan_local,
    )
    print("frame meta:", meta)

    pts = load_lidar_bin(lidar_path, num_cols=len(LIDAR_COLS))
    coord = pts[:, :3]
    intensity = pts[:, 3]
    mask = point_clip_mask(coord, args.pc_range)
    coord = coord[mask]
    intensity = intensity[mask]
    # Waymo preprocess: strength = tanh(intensity). Robotruck intensity is 0–255.
    strength = np.tanh(intensity.reshape(-1, 1) / 255.0).astype(np.float32)
    print(f"points after clip: {coord.shape[0]} (from {pts.shape[0]})")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_segmentor(ROOT / args.config_file, ROOT / args.weight, device)
    pred = infer_frame(model, coord, strength, device, grid_size=args.grid_size)
    assert pred.shape[0] == coord.shape[0]

    stem = f"{meta.get('clip_name') or meta['clip_id']}_{meta['timestamp']}"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    pred_path = out_dir / f"{stem}_pred.npy"
    np.save(pred_path, pred.astype(np.int32))
    coord_path = out_dir / f"{stem}_coord.npy"
    np.save(coord_path, coord.astype(np.float32))

    vis_path = out_dir / f"{stem}_pred_vis.png"
    render_pred_vis(
        coord,
        pred,
        intensity,
        vis_path,
        title=f"Robotruck {meta.get('clip_name')} ts={meta['timestamp']} md5={meta['md5'][:8]}",
        max_points=args.max_points_vis,
        seed=args.seed,
    )
    legend_path = out_dir / "waymo_class_legend.png"
    save_class_legend(legend_path)

    print(f"pred -> {pred_path}")
    print(f"vis  -> {vis_path}")
    print(f"legend -> {legend_path}")
    print("note: no dense semseg GT in this Robotruck frame collection; vis is Pred vs Intensity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
