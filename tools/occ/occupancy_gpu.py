"""Exact integer voxel counts and class votes on CUDA; never sample inputs."""
from threading import Lock, local

import numpy as np

_kernel = None
_compile_lock = Lock()
_streams = local()
_SOURCE = r'''
extern "C" __global__ void vote(
    const float* xyz, const int* labels, long long n,
    float x0, float x1, float y0, float y1, float z0, float z1,
    float voxel, int nx, int ny, int nz, int classes,
    int* counts, int* votes, float* heights) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = xyz[3*i], y = xyz[3*i+1], z = xyz[3*i+2];
    if (!(x >= x0 && x < x1 && y >= y0 && y < y1 && z >= z0 && z < z1)) return;
    int ix = min(nx-1, max(0, (int)floorf((x-x0)/voxel)));
    int iy = min(ny-1, max(0, (int)floorf((y-y0)/voxel)));
    int iz = min(nz-1, max(0, (int)floorf((z-z0)/voxel)));
    long long key = ix + (long long)nx * (iy + (long long)ny * iz);
    atomicAdd(counts + key, 1);
    int label = labels[i];
    if (label >= 0) atomicAdd(votes + key * classes + label, 1);
    int* address = (int*)(heights + key);
    int old = *address, assumed;
    while (__int_as_float(old) < z) {
        assumed = old;
        old = atomicCAS(address, assumed, __float_as_int(z));
        if (old == assumed) break;
    }
}
'''


def components(xyz, labels, bounds, voxel, shape, min_points):
    import cupy as cp
    global _kernel
    with _compile_lock:
        if _kernel is None:
            _kernel = cp.RawKernel(_SOURCE, 'vote', options=('--fmad=false', '--prec-div=true'))
            _kernel.compile()
    classes = max(1, int(labels.max()) + 1)
    cells = int(np.prod(shape))
    # One stream per frame allows independent frame workers to overlap copies
    # and kernels. All returned values have been copied back before it exits.
    if not hasattr(_streams, 'stream'):
        _streams.stream = cp.cuda.Stream(non_blocking=True)
    with _streams.stream:
        xyz_gpu = cp.ascontiguousarray(cp.asarray(xyz, dtype=cp.float32))
        labels_gpu = cp.ascontiguousarray(cp.asarray(labels, dtype=cp.int32))
        counts = cp.zeros(cells, dtype=cp.int32)
        votes = cp.zeros((cells, classes), dtype=cp.int32)
        heights = cp.full(cells, -cp.inf, dtype=cp.float32)
        args = (xyz_gpu, labels_gpu, np.int64(len(xyz)),
                *(np.float32(v) for v in bounds), np.float32(voxel),
                *(np.int32(v) for v in shape), np.int32(classes), counts, votes, heights)
        _kernel(((len(xyz) + 255) // 256,), (256,), args)
        keys = cp.flatnonzero(counts >= max(1, int(min_points)))
        nx, ny, _ = shape
        ijk = cp.stack((keys % nx, (keys // nx) % ny, keys // (nx * ny)), axis=1).astype(cp.int32)
        return (ijk.get(), votes[keys].argmax(axis=1).astype(cp.int32).get(),
                counts[keys].get(), heights[keys].get())


def frame_stream():
    import cupy as cp
    if not hasattr(_streams, 'stream'):
        _streams.stream = cp.cuda.Stream(non_blocking=True)
    return _streams.stream


def build_grid(xyz, labels, *, x_range, y_range, z_range, voxel, min_points=1):
    from occupancy import OccupancyGrid, _range_to_size
    shape = tuple(_range_to_size(lo, hi, voxel) for lo, hi in (x_range, y_range, z_range))
    if len(xyz) == 0:
        return OccupancyGrid(np.empty((0,3),np.int32), np.empty((0,3),np.float32),
            np.empty(0,np.int32), np.empty(0,np.int32), np.empty(0,np.float32),
            voxel, x_range, y_range, z_range, shape)
    ijk, majority, counts, max_z = components(
        xyz, labels, (*x_range, *y_range, *z_range), voxel, shape, min_points)
    centers = np.stack([lo + (ijk[:, axis].astype(np.float32) + .5) * voxel
                        for axis, lo in enumerate((x_range[0], y_range[0], z_range[0]))], axis=1).astype(np.float32)
    return OccupancyGrid(ijk, centers, majority, counts, max_z, voxel,
                         x_range, y_range, z_range, shape)
