"""Prefer native operators; retain the small inference subset when absent.

Only a missing top-level package selects the fallback. Broken native installs
and missing transitive dependencies remain errors rather than changing backend.
"""
from importlib import import_module


def __getattr__(name):
    if name not in ('flash_attn', 'torch_scatter', 'pointops'):
        raise AttributeError(name)
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return import_module('.' + name, __name__)
