# LitePT CUDA env fix (HAMI vGPU + nvidia1-only container + compat-miss)
# Source this at the top of every shell that runs CUDA code for LitePT.
#
# Fixes applied:
#   1. CUDA_VISIBLE_DEVICES=1      – /dev/nvidia0 is missing; only nvidia1 exists
#   2. Unset _CUDA_COMPAT_PATH     – forced compat libs are OLDER than the
#                                    running driver (575 vs 580) and cause
#                                    cuInit error 304
#   3. Strip Windows Path spillover
#   4. Preload DRIVER libcuda.so via LD_PRELOAD/ld.so.cache, not compat
#   5. HAMI vGPU runtime stabilisers

export CUDA_VISIBLE_DEVICES=0
unset _CUDA_COMPAT_PATH
# Remove the case-insensitive "Path" that leaked from some desktop/sandbox.
unset Path 2>/dev/null || true

# Prefer the driver-provided libcuda (matches nvidia-smi driver version) over
# any compat build.  ldconfig already has /usr/lib/x86_64-linux-gnu cached
# but we make it explicit so subprocess children inherit the pick reliably.
if [ -z "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib"
else
  case ":$LD_LIBRARY_PATH:" in
    *":/usr/lib/x86_64-linux-gnu:"*) ;;
    *) export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" ;;
  esac
fi

# HAMI vGPU known env flags for avoiding cuInit 304 in subprocess-heavy loads.
export HAMI_DISABLE_WARN=1          # silence recursive-dlsym chatter (harmless but pollutes logs)

# PyTorch: avoid libcudnn/driver auto-upgrade path when compat env was set before shell.
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
# Prevent CUDA from trying to fall back to JIT patching which races on HAMI.
export CUDA_MODULE_LOADING=EAGER
