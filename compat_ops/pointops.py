"""Inference-only placeholder for LitePT's optional evaluator extension.

The semantic-segmentation production path does not call PointOps.  Keeping
this module importable prevents evaluator registration from forcing a local
CUDA build on machines where no prebuilt wheel is available.
"""


def __getattr__(name):
    raise AttributeError(f'PointOps {name!r} requires the native pointops extension; '
                         'the compatibility module only supports inference imports')
