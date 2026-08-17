"""Render Waymo LitePT pred vs GT as BEV / side-view PNGs (no display needed)."""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Waymo 22-class palette (RGB 0-1), aligned with configs/waymo names
WAYMO_NAMES = [
    "Car",
    "Truck",
    "Bus",
    "Other Vehicle",
    "Motorcyclist",
    "Bicyclist",
    "Pedestrian",
    "Sign",
    "Traffic Light",
    "Pole",
    "Construction Cone",
    "Bicycle",
    "Motorcycle",
    "Building",
    "Vegetation",
    "Tree Trunk",
    "Curb",
    "Road",
    "Lane Marker",
    "Other Ground",
    "Walkable",
    "Sidewalk",
]
WAYMO_COLORS = np.array(
    [
        [0.90, 0.10, 0.10],  # 0 Car
        [0.80, 0.20, 0.20],  # 1 Truck
        [0.70, 0.10, 0.30],  # 2 Bus
        [0.85, 0.40, 0.15],  # 3 Other Vehicle
        [1.00, 0.50, 0.00],  # 4 Motorcyclist
        [1.00, 0.70, 0.00],  # 5 Bicyclist
        [1.00, 0.00, 0.80],  # 6 Pedestrian
        [1.00, 1.00, 0.00],  # 7 Sign
        [1.00, 0.85, 0.20],  # 8 Traffic Light
        [0.60, 0.40, 0.20],  # 9 Pole
        [1.00, 0.60, 0.00],  # 10 Construction Cone
        [0.20, 0.80, 0.20],  # 11 Bicycle
        [0.10, 0.60, 0.10],  # 12 Motorcycle
        [0.55, 0.55, 0.55],  # 13 Building
        [0.10, 0.70, 0.20],  # 14 Vegetation
        [0.40, 0.25, 0.10],  # 15 Tree Trunk
        [0.70, 0.70, 0.40],  # 16 Curb
        [0.35, 0.35, 0.40],  # 17 Road
        [0.95, 0.95, 0.95],  # 18 Lane Marker
        [0.50, 0.45, 0.30],  # 19 Other Ground
        [0.60, 0.80, 0.90],  # 20 Walkable
        [0.70, 0.70, 0.85],  # 21 Sidewalk
    ],
    dtype=np.float32,
)
IGNORE_COLOR = np.array([0.18, 0.18, 0.18], dtype=np.float32)
OK_COLOR = np.array([0.15, 0.85, 0.25], dtype=np.float32)
BAD_COLOR = np.array([0.95, 0.15, 0.15], dtype=np.float32)


def save_class_legend(out_png: Path) -> None:
    """Write a standalone Waymo class / compare-color legend PNG."""
    ncols = 2
    nrows = (len(WAYMO_NAMES) + ncols - 1) // ncols
    fig_h = 0.55 * nrows + 1.8
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    ax.text(
        0.5,
        0.98,
        "Waymo Semantic Classes · LitePT vis palette",
        ha="center",
        va="top",
        color="white",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )

    top = 0.90
    row_h = 0.78 / nrows
    for i, name in enumerate(WAYMO_NAMES):
        r, c = divmod(i, ncols)
        x0 = 0.06 + c * 0.48
        y = top - r * row_h - 0.02
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y - 0.028),
                0.055,
                0.038,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor=WAYMO_COLORS[i],
                edgecolor="#666666",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(
            x0 + 0.07,
            y - 0.008,
            f"{i:2d}  {name}",
            ha="left",
            va="center",
            color="white",
            fontsize=11,
            transform=ax.transAxes,
        )

    extras = [
        ("ignore (-1)", IGNORE_COLOR),
        ("agree (pred==gt)", OK_COLOR),
        ("error (pred!=gt)", BAD_COLOR),
    ]
    y_extra = 0.08
    ax.text(
        0.06,
        y_extra + 0.04,
        "Compare panel colors",
        ha="left",
        va="bottom",
        color="#aaaaaa",
        fontsize=10,
        transform=ax.transAxes,
    )
    for j, (lab, col) in enumerate(extras):
        x0 = 0.06 + j * 0.31
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y_extra - 0.02),
                0.045,
                0.032,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor=col,
                edgecolor="#666666",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(
            x0 + 0.055,
            y_extra - 0.004,
            lab,
            ha="left",
            va="center",
            color="white",
            fontsize=10,
            transform=ax.transAxes,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

PRED_RE = re.compile(
    r"^(?P<seg>segment-.+_with_camera_labels)_(?P<ts>\d+)_pred\.npy$"
)


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.tile(IGNORE_COLOR, (labels.shape[0], 1))
    valid = (labels >= 0) & (labels < len(WAYMO_COLORS))
    colors[valid] = WAYMO_COLORS[labels[valid]]
    return colors


def agreement_colors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    colors = np.tile(IGNORE_COLOR, (pred.shape[0], 1))
    labeled = gt >= 0
    colors[labeled & (pred == gt)] = OK_COLOR
    colors[labeled & (pred != gt)] = BAD_COLOR
    return colors


def resolve_frame_dir(pred_path: Path, data_root: Path) -> Path | None:
    m = PRED_RE.match(pred_path.name)
    if not m:
        return None
    seg, ts = m.group("seg"), m.group("ts")
    frame_dir = data_root / "validation" / seg / ts
    if (frame_dir / "coord.npy").is_file() and (frame_dir / "segment.npy").is_file():
        return frame_dir
    return None


def _style_ax(ax) -> None:
    ax.set_facecolor("black")
    ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#444444")


def scatter_xy(ax, xyz: np.ndarray, colors: np.ndarray, title: str, axes: str) -> None:
    if axes == "xy":
        ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=0.12, linewidths=0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.scatter(xyz[:, 0], xyz[:, 2], c=colors, s=0.12, linewidths=0)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    ax.set_title(title)
    _style_ax(ax)


def render_compare(
    coords: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_png: Path,
    title: str,
    max_points: int,
    seed: int,
) -> dict:
    labeled = gt >= 0
    n_lab = int(labeled.sum())
    acc = float((pred[labeled] == gt[labeled]).mean()) if n_lab else float("nan")

    n = coords.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        coords = coords[idx]
        pred = pred[idx]
        gt = gt[idx]

    pred_c = labels_to_colors(pred)
    gt_c = labels_to_colors(gt)
    diff_c = agreement_colors(pred, gt)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=130)
    scatter_xy(axes[0, 0], coords, pred_c, "LitePT Pred · BEV", "xy")
    scatter_xy(axes[0, 1], coords, gt_c, "GT · BEV", "xy")
    scatter_xy(axes[0, 2], coords, diff_c, "Agree(green)/Err(red)/Ign(gray) · BEV", "xy")
    scatter_xy(axes[1, 0], coords, pred_c, "LitePT Pred · Side", "xz")
    scatter_xy(axes[1, 1], coords, gt_c, "GT · Side", "xz")
    scatter_xy(axes[1, 2], coords, diff_c, "Agree/Err/Ign · Side", "xz")

    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"{title}   |   labeled_acc={acc:.3f}  (N_labeled={n_lab}/{n})",
        color="white",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"acc": acc, "n_labeled": n_lab, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", default="exp/waymo/single_clip_eval/result")
    ap.add_argument("--data-root", default="data/waymo")
    ap.add_argument(
        "--clip",
        default="segment-10203656353524179475_7625_000_7645_000_with_camera_labels",
        help="Only visualize this clip; empty = all preds",
    )
    ap.add_argument("--out-dir", default="exp/waymo/single_clip_eval/vis")
    ap.add_argument("--max-frames", type=int, default=5)
    ap.add_argument("--max-points", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--legend-only",
        action="store_true",
        help="Only write waymo_class_legend.png and exit",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    result_dir = (root / args.result_dir).resolve()
    data_root = (root / args.data_root).resolve()
    out_dir = (root / args.out_dir).resolve()

    legend_path = out_dir / "waymo_class_legend.png"
    save_class_legend(legend_path)
    print(f"legend -> {legend_path}")
    if args.legend_only:
        return 0

    pattern = f"{args.clip}_*_pred.npy" if args.clip else "*_pred.npy"
    pred_files = sorted(result_dir.glob(pattern))
    if not pred_files:
        pred_files = [Path(p) for p in sorted(glob.glob(str(result_dir / "*_pred.npy")))]

    if not pred_files:
        print(f"No _pred.npy under {result_dir}")
        return 1

    pred_files = pred_files[: args.max_frames]
    print(f"Rendering {len(pred_files)} frame(s) Pred|GT|Diff -> {out_dir}")

    for i, pred_path in enumerate(pred_files):
        pred_path = Path(pred_path)
        pred = np.load(pred_path).astype(np.int64).reshape(-1)
        frame_dir = resolve_frame_dir(pred_path, data_root)
        if frame_dir is None:
            print(f"skip (no frame dir): {pred_path.name}")
            continue
        coords = np.load(frame_dir / "coord.npy").astype(np.float32)
        gt = np.load(frame_dir / "segment.npy").astype(np.int64).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] < 3:
            print(f"skip (bad coord): {frame_dir}")
            continue
        if not (coords.shape[0] == pred.shape[0] == gt.shape[0]):
            print(
                f"skip (N mismatch): pred={pred.shape[0]} gt={gt.shape[0]} "
                f"coord={coords.shape[0]} {pred_path.name}"
            )
            continue

        out_png = out_dir / pred_path.name.replace("_pred.npy", "_pred_vs_gt.png")
        title = pred_path.name.replace("_pred.npy", "")
        stats = render_compare(
            coords[:, :3],
            pred,
            gt,
            out_png,
            title=title,
            max_points=args.max_points,
            seed=args.seed + i,
        )
        print(
            f"OK {out_png.name}  labeled_acc={stats['acc']:.3f}  "
            f"N_lab={stats['n_labeled']}/{stats['n']}"
        )

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
