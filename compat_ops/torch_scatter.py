"""Small pure-PyTorch subset used by LitePT inference when no wheel exists."""
import torch


def segment_csr(src, indptr, reduce='sum'):
    if reduce not in ('sum', 'mean', 'max'):
        raise ValueError(f'unsupported reduction: {reduce}')
    # Pool all clusters in one prebuilt CUDA operation, not one kernel per point.
    result = torch.segment_reduce(src, reduce, offsets=indptr, axis=0)
    empty = indptr[1:] == indptr[:-1]
    return result.masked_fill(empty.reshape((-1,) + (1,) * (src.ndim - 1)), 0)


def scatter_add(src, index, dim=0, dim_size=None):
    if dim != 0:
        raise ValueError('fallback supports dim=0 only')
    size = int(index.max()) + 1 if dim_size is None and index.numel() else int(dim_size or 0)
    output = src.new_zeros((size, *src.shape[1:]))
    output.index_add_(0, index, src)
    return output


def scatter_mean(src, index, dim=0, dim_size=None):
    total = scatter_add(src, index, dim=dim, dim_size=dim_size)
    ones = src.new_ones((src.shape[0],))
    counts = scatter_add(ones, index, dim=0, dim_size=total.shape[0]).clamp_min_(1)
    return total / counts.reshape((-1,) + (1,) * (src.ndim - 1))
