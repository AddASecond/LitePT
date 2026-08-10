#!/usr/bin/env python3
"""Run Waymo LitePT-S semantic segmentation on 1–few converted smoke clouds.

18GB-oriented defaults: batch=1, AMP, GridSample 0.05, optional SphereCrop via
pre-subsample in convert script. Loads HF Waymo checkpoint and prints logits.

Usage:
  python tools/convert_few_clouds_for_smoke.py
  python tools/smoke_seg_infer.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# repo root on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import types  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _install_optional_stubs() -> None:
    """Stub optional CUDA-only deps so smoke works without full LitePT build."""
    if "pointops" not in sys.modules:
        sys.modules["pointops"] = types.ModuleType("pointops")

    try:
        import flash_attn  # noqa: F401
    except Exception:
        mod = types.ModuleType("flash_attn")

        def flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=0.0,
            softmax_scale=None,
            **kwargs,
        ):
            # qkv: [N, 3, H, D] packed by variable-length sequences in cu_seqlens
            outs = []
            cu = cu_seqlens.tolist()
            drop = float(dropout_p or 0.0)
            for i in range(len(cu) - 1):
                s, e = int(cu[i]), int(cu[i + 1])
                qkv_i = qkv[s:e]
                q = qkv_i[:, 0].transpose(0, 1).unsqueeze(0)
                k = qkv_i[:, 1].transpose(0, 1).unsqueeze(0)
                v = qkv_i[:, 2].transpose(0, 1).unsqueeze(0)
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=drop if qkv.requires_grad else 0.0, scale=softmax_scale
                )
                outs.append(out.squeeze(0).transpose(0, 1).reshape(e - s, -1))
            return torch.cat(outs, dim=0)

        mod.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
        sys.modules["flash_attn"] = mod
        print("[smoke] using SDPA fallback for flash_attn")


_install_optional_stubs()

from datasets.transform import Compose  # noqa: E402
from models.builder import build_model  # noqa: E402
import models.litept  # noqa: E402,F401  — register LitePT backbone
import models.default  # noqa: E402,F401 — register DefaultSegmentorV2
import models.losses  # noqa: E402,F401


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


def build_waymo_segmentor():
    cfg = dict(
        type="DefaultSegmentorV2",
        num_classes=22,
        backbone_out_channels=72,
        backbone=dict(
            type="LitePT",
            in_channels=4,
            order=["z", "z-trans", "hilbert", "hilbert-trans"],
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(36, 72, 144, 252, 504),
            enc_num_head=(2, 4, 8, 14, 28),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            enc_conv=(True, True, True, False, False),
            enc_attn=(False, False, False, True, True),
            enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
            dec_depths=(0, 0, 0, 0),
            dec_channels=(72, 72, 144, 252),
            dec_num_head=(4, 4, 8, 14),
            dec_patch_size=(1024, 1024, 1024, 1024),
            dec_conv=(False, False, False, False),
            dec_attn=(False, False, False, False),
            dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enc_mode=False,
        ),
        criteria=[
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        ],
    )
    return build_model(cfg)


def load_waymo_weights(model: torch.nn.Module, ckpt_path: str | None) -> None:
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id="prs-eth/LitePT",
            filename="waymo-semseg-litept-small-v1m1/model/model_best.pth",
            repo_type="model",
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    weight = OrderedDict()
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        weight[new_key] = value
    missing, unexpected = model.load_state_dict(weight, strict=False)
    print(f"loaded {ckpt_path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")


def list_frames(data_root: Path) -> list[Path]:
    frames = sorted(p for p in data_root.glob("training/*/*") if p.is_dir())
    return frames


def load_frame(frame_dir: Path) -> dict:
    coord = np.load(frame_dir / "coord.npy").astype(np.float32)
    strength = np.load(frame_dir / "strength.npy").astype(np.float32)
    if strength.ndim == 1:
        strength = strength.reshape(-1, 1)
    return dict(coord=coord, strength=strength, name=str(frame_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default=str(ROOT / "data" / "waymo_smoke"),
    )
    ap.add_argument("--weight", default=None, help="Optional local .pth; else HF Waymo")
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-pred", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="Disable autocast (safer on odd GPUs)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    frames = list_frames(data_root)
    if not frames:
        raise SystemExit(
            f"No frames under {data_root}. Run: python tools/convert_few_clouds_for_smoke.py"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  total_mem_GB={props.total_memory / 1024**3:.1f}")

    model = build_waymo_segmentor()
    load_waymo_weights(model, args.weight)

    # Prebuilt spconv-cu124 may fail ImplicitGemm algo select on B200/sm_100
    # (and when HAMI reports total_memory=0). Native algo is slower but works.
    try:
        import spconv.pytorch as spconv
        from spconv.core import ConvAlgo

        patched = 0
        for m in model.modules():
            if isinstance(
                m,
                (
                    spconv.SubMConv3d,
                    spconv.SparseConv3d,
                    spconv.SparseInverseConv3d,
                ),
            ):
                m.algo = ConvAlgo.Native
                patched += 1
        if patched:
            print(f"spconv: forced ConvAlgo.Native on {patched} modules")
    except Exception as e:
        print(f"spconv algo patch skipped: {e}")

    model.to(device)
    model.eval()

    transform = Compose(
        [
            dict(
                type="GridSample",
                grid_size=args.grid_size,
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

    for frame_dir in frames:
        raw = load_frame(frame_dir)
        n0 = raw["coord"].shape[0]
        point = transform(raw)
        with torch.no_grad():
            use_amp = device.type == "cuda" and not args.no_amp
            amp_ctx = (
                torch.amp.autocast("cuda", enabled=use_amp)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with amp_ctx:
                for key, val in list(point.items()):
                    if isinstance(val, torch.Tensor):
                        point[key] = val.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                out = model(point)
                logits = out["seg_logits"]  # [N_voxel, 22]
                # map back to original points via inverse from GridSample
                inverse = point["inverse"]
                dense_logits = logits[inverse]
                pred = dense_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)

        peak = (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        )
        # class histogram ignoring empty
        hist = np.bincount(pred, minlength=22)
        top = sorted(enumerate(hist), key=lambda x: -x[1])[:5]
        top_s = ", ".join(f"{WAYMO_NAMES[i]}:{c}" for i, c in top if c > 0)
        print(
            f"OK {frame_dir.name}  N={n0}  voxels={logits.shape[0]}  "
            f"logits={tuple(dense_logits.shape)}  peak_vram_GB={peak:.2f}"
        )
        print(f"  top classes: {top_s}")

        if args.save_pred:
            np.save(frame_dir / "pred_segment.npy", pred)
            print(f"  saved {frame_dir / 'pred_segment.npy'}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("smoke segmentation inference succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
