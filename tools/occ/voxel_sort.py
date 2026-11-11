"""Optional GPU integer-key sorting; geometry and voxel arithmetic stay unchanged."""
import os

import numpy as np

_cupy = None
_unavailable = False


def argsort(keys):
    global _cupy, _unavailable
    backend = os.environ.get('OCC_VOXEL_SORT', 'auto')
    if backend not in ('auto', 'cpu', 'gpu'):
        raise ValueError(f'Unknown OCC_VOXEL_SORT: {backend}')
    if backend == 'cpu' or len(keys) < 8192 or _unavailable:
        return np.argsort(keys)
    try:
        if _cupy is None:
            import cupy
            _cupy = cupy
        return _cupy.argsort(_cupy.asarray(keys)).get()
    except (ImportError, RuntimeError, OSError) as error:
        if backend == 'gpu':
            raise
        _unavailable = True
        print(f'OCC GPU sorting unavailable; using NumPy: {error}', flush=True)
        return np.argsort(keys)
