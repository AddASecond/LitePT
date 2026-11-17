#!/usr/bin/env python3
"""Keep one A-lane inference model alive behind a local, sequential CLI worker."""
import argparse
import faulthandler
import gc
import json
import os
import socketserver
import time
import traceback
from pathlib import Path


def release_idle_memory():
    # Keep the model, but return freed preprocessing buffers to the OS between
    # clips instead of retaining an idle glibc heap.
    gc.collect()
    try:
        import ctypes
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)


def serve(path):
    # Loading only kernels actually used by this model avoids several GiB of
    # host-side CUDA modules. An explicit environment setting still wins.
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
    import warmup_robotruck_preds
    model_cache = {}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            started = time.monotonic()
            faulthandler.dump_traceback_later(60, repeat=True)
            try:
                request = json.loads(self.rfile.readline(4 * 1024 * 1024))
                print(f'Prediction request started: {request["argv"][:4]}', flush=True)
                code = warmup_robotruck_preds.main(request['argv'], model_cache=model_cache)
                response = {'ok': code == 0}
            except (Exception, SystemExit):
                error = traceback.format_exc()
                print(error, flush=True)
                response = {'ok': False, 'error': error}
            finally:
                faulthandler.cancel_dump_traceback_later()
                release_idle_memory()
            response['elapsed_seconds'] = time.monotonic() - started
            self.wfile.write((json.dumps(response) + '\n').encode())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Prediction worker socket already exists: {path}')
    try:
        with socketserver.UnixStreamServer(str(path), Handler) as server:
            os.chmod(path, 0o600)
            print(f'Prediction worker ready: {path}', flush=True)
            server.serve_forever()
    finally:
        path.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', required=True)
    serve(parser.parse_args().socket)
