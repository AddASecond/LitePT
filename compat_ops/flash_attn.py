"""Prebuilt-wheel-free compatibility subset of :mod:`flash_attn`.

The production model only needs varlen packed QKV attention.  PyTorch's CUDA
scaled-dot-product attention selects its own fused kernel when available, so
this keeps inference functional on machines without a matching flash-attn
wheel and never requires compiling an extension.
"""
import torch
import torch.nn.functional as F


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                                     softmax_scale=None):
    """Apply independent self-attention to packed ``[N, 3, H, D]`` segments."""
    del max_seqlen
    boundaries = cu_seqlens.detach().to(device='cpu', dtype=torch.int64).tolist()
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if lengths and lengths[0] > 0 and all(n == lengths[0] for n in lengths):
        packed = qkv.reshape(len(lengths), lengths[0], *qkv.shape[1:])
        q, k, v = (packed[:, :, i].transpose(1, 2) for i in range(3))
        value = F.scaled_dot_product_attention(
            q, k, v, dropout_p=float(dropout_p), is_causal=False, scale=softmax_scale
        )
        return value.transpose(1, 2).reshape(-1, qkv.shape[2], qkv.shape[3])
    output = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end == start:
            continue
        segment = qkv[start:end]
        q, k, v = (segment[:, i].transpose(0, 1).unsqueeze(0) for i in range(3))
        value = F.scaled_dot_product_attention(
            q, k, v, dropout_p=float(dropout_p), is_causal=False, scale=softmax_scale
        )
        output.append(value.squeeze(0).transpose(0, 1))
    if output:
        return torch.cat(output, dim=0)
    return qkv.new_empty((0, qkv.shape[2], qkv.shape[3]))
