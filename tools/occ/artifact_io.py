"""Bounded streaming of immutable Scene bytes to the existing S3 asset format."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Condition
import time

_active_sink = None


def write_bytes(path, data):
    path.write_bytes(data)
    if _active_sink is not None:
        _active_sink.submit(path, data=data)


def camera_ready(path, md5):
    if _active_sink is not None:
        _active_sink.submit(path, source_camera_md5=md5)


@contextmanager
def use_sink(sink):
    global _active_sink
    if _active_sink is not None:
        raise RuntimeError('An artifact sink is already active in this exporter')
    _active_sink = sink
    try:
        yield
    finally:
        _active_sink = None
        if sink is not None:
            sink.close()


class AssetPipeline:
    """Upload while computing; expose refs only after every requested asset passes."""
    def __init__(self, publisher, scene, *, timestamps=None, workers=16, max_bytes=128 * 1024**2):
        self.publisher = publisher
        self.scene = Path(scene).resolve()
        self.timestamps = None if timestamps is None else set(map(str, timestamps))
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.condition = Condition()
        self.max_bytes = max_bytes
        self.pending_bytes = 0
        self.peak_pending_bytes = 0
        self.futures = {}
        self.signatures = {}
        self.failure = None
        self.started = time.monotonic()
        self.closed = False

    @staticmethod
    def signature(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def submit(self, path, *, data=None, source_camera_md5=None):
        path = Path(path).absolute()
        relative = path.relative_to(self.scene)
        if (self.timestamps is not None and relative.parts[0] == 'frames'
                and relative.parts[1] not in self.timestamps):
            return
        path = path.resolve()
        size = path.stat().st_size if data is None else len(data)
        with self.condition:
            while self.pending_bytes and self.pending_bytes + size > self.max_bytes and self.failure is None:
                self.condition.wait()
            if self.failure is not None:
                raise RuntimeError('Asynchronous asset publication failed') from self.failure
            if self.closed:
                raise RuntimeError('Artifact pipeline is closed')
            signature = self.signature(path)
            if path in self.futures:
                if self.signatures[path] != signature:
                    raise ValueError(f'Asset changed during publication: {path}')
                return
            self.pending_bytes += size
            self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
            self.signatures[path] = signature
            future = self.pool.submit(self.publisher, path, data=data, source_camera_md5=source_camera_md5)
            self.futures[path] = future

        def completed(future):
            with self.condition:
                self.pending_bytes -= size
                if future.exception() is not None:
                    self.failure = future.exception()
                self.condition.notify_all()
        future.add_done_callback(completed)

    def publish(self, path):
        path = Path(path).resolve()
        if path not in self.futures:
            raise ValueError(f'Asset was not produced by this export: {path}')
        if self.signature(path) != self.signatures[path]:
            raise ValueError(f'Asset changed after publication: {path}')
        return dict(self.futures[path].result())

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True)
            self.closed = True
        if self.failure is not None:
            raise RuntimeError('Asynchronous asset publication failed') from self.failure

    def stats(self):
        return {'assets': len(self.futures), 'peak_pending_bytes': self.peak_pending_bytes,
                'elapsed_seconds': time.monotonic() - self.started}
