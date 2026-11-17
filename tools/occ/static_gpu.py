"""Keep immutable clip points on CUDA while transforming each vehicle view."""
from threading import Lock

import numpy as np

_lock = Lock()
_transform = None
_SOURCE = r'''
extern "C" __global__ void transform(const float* points, const double* matrix,
                                    float* output, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = points[3*i], y = points[3*i+1], z = points[3*i+2];
    for (int j=0; j<3; ++j) {
        const double* row = matrix + 4*j;
        double value = __fma_rn(row[0], x, 0.0);
        value = __fma_rn(row[1], y, value);
        value = __fma_rn(row[2], z, value);
        value = __fma_rn(row[3], 1.0, value);
        output[3*i+j] = (float)value;
    }
}
'''


def prepare(agg):
    import cupy as cp
    global _transform
    with _lock:
        if _transform is None:
            _transform = cp.RawKernel(_SOURCE, 'transform', options=('--fmad=false',))
            _transform.compile()
        if '_cuda_static' not in agg:
            if len(agg['labels']) and (np.min(agg['labels']) < 0 or np.max(agg['labels']) >= 22):
                raise ValueError('Expected Waymo-22 static labels')
            # Aggregation's transposed float64 transform yields Fortran-order
            # XYZ. The raw kernel indexes packed XYZ triples, so normalize the
            # layout explicitly instead of preserving NumPy's memory order.
            arrays = (cp.ascontiguousarray(cp.asarray(agg['xyz_map'], dtype=cp.float32)),
                      cp.ascontiguousarray(cp.asarray(agg['labels'], dtype=cp.int32)))
            cp.cuda.get_current_stream().synchronize()
            agg['_cuda_static'] = arrays
    return agg['_cuda_static']


def view(agg, T_map_vehicle, *, x_range, y_range, z_range):
    import cupy as cp
    points, labels = prepare(agg)
    matrix = cp.ascontiguousarray(cp.asarray(np.linalg.inv(T_map_vehicle), dtype=cp.float64))
    xyz = cp.empty_like(points)
    _transform(((len(points)+255)//256,), (256,), (points, matrix, xyz, np.int64(len(points))))
    keep = cp.ones(len(points), dtype=cp.bool_)
    for axis, bounds in enumerate((x_range, y_range, z_range)):
        keep &= (xyz[:, axis] >= bounds[0]) & (xyz[:, axis] <= bounds[1])
    return xyz[keep], labels[keep]
