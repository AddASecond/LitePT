"""Shared CUDA / HAMI process environment for OCC export runners."""
from __future__ import annotations

import os


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
