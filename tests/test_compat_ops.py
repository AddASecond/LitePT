"""Portable inference operators preserve segment boundaries and empty groups."""
import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compat_ops


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('lengths', [[3, 3], [2, 0, 5], []])
def test_attention_matches_independent_per_segment_softmax(device, lengths):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    fallback = importlib.import_module('compat_ops.flash_attn')
    torch.manual_seed(17)
    packed = torch.randn(sum(lengths), 3, 2, 4, device=device)
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    actual = fallback.flash_attn_varlen_qkvpacked_func(packed, boundaries, max(lengths, default=0), softmax_scale=.3)
    expected = []
    for start, end in zip(boundaries.tolist()[:-1], boundaries.tolist()[1:]):
        q, k, v = [packed[start:end, i].transpose(0, 1) for i in range(3)]
        expected.append(((q @ k.transpose(-1, -2)*.3).softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(expected) if expected else packed.new_empty((0, 2, 4))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_segment_and_scatter_reductions_with_empty_groups(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    fallback = importlib.import_module('compat_ops.torch_scatter')
    values = torch.tensor([[1., -2.], [3., 4.], [-1., 6.]], device=device)
    offsets = torch.tensor([0, 2, 2, 3], device=device)
    for reduction, expected in [('sum', [[4, 2], [0, 0], [-1, 6]]),
                                 ('mean', [[2, 1], [0, 0], [-1, 6]]),
                                 ('max', [[3, 4], [0, 0], [-1, 6]])]:
        torch.testing.assert_close(fallback.segment_csr(values, offsets, reduction), values.new_tensor(expected))
    index = torch.tensor([2, 0, 2], device=device)
    torch.testing.assert_close(fallback.scatter_add(values, index, dim_size=4), values.new_tensor([[3, 4], [0, 0], [0, 4], [0, 0]]))
    torch.testing.assert_close(fallback.scatter_mean(values, index, dim_size=4), values.new_tensor([[3, 4], [0, 0], [0, 2], [0, 0]]))
    assert fallback.scatter_add(values[:0], index[:0], dim_size=4).shape == (4, 2)


def test_native_precedence_and_broken_install_is_not_hidden(monkeypatch):
    native = object()
    monkeypatch.setattr(compat_ops, 'import_module', lambda name, *args: native)
    assert compat_ops.__getattr__('flash_attn') is native
    def broken(name, *args):
        raise ModuleNotFoundError('native dependency missing', name='flash_attn_2_cuda')
    monkeypatch.setattr(compat_ops, 'import_module', broken)
    with pytest.raises(ModuleNotFoundError, match='native dependency'):
        compat_ops.__getattr__('flash_attn')


def test_missing_package_uses_versioned_fallback(monkeypatch):
    original = importlib.import_module
    def missing(name, *args):
        if name == 'torch_scatter':
            raise ModuleNotFoundError(name=name)
        return original(name, *args)
    monkeypatch.setattr(compat_ops, 'import_module', missing)
    assert compat_ops.__getattr__('torch_scatter').__name__ == 'compat_ops.torch_scatter'
    pointops = original('compat_ops.pointops')
    with pytest.raises(AttributeError, match='native pointops'):
        pointops.knn_query
