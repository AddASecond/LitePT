"""Selected IMU trajectories shared by scan deskew, aggregation and cameras."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class PoseSamples(list):
    """List-compatible samples with bounded, vectorized nanosecond interpolation."""

    def __init__(self, samples, *, max_gap_ns=250_000_000, endpoint_tolerance_ns=10_000_000):
        super().__init__(samples)
        self._pose_cache = {}
        self._matrix_cache = {}
        self.times = np.asarray([int(t) for t, _ in samples], dtype=np.int64)
        self.positions = np.asarray([[p['position'][k] for k in 'xyz'] for _, p in samples], dtype=np.float64)
        quats = np.asarray([[p['orientation'][k] for k in 'xyzw'] for _, p in samples], dtype=np.float64)
        if (len(self.times) < 2 or np.any(np.diff(self.times) <= 0)
                or not np.isfinite(self.positions).all() or not np.isfinite(quats).all()
                or np.any(np.linalg.norm(quats, axis=1) < 1e-6)):
            raise ValueError('Invalid or unordered selected pose samples')
        self.quaternions = Rotation.from_quat(quats).as_quat()
        self.max_gap_ns = int(max_gap_ns)
        self.endpoint_tolerance_ns = int(endpoint_tolerance_ns)

    def poses_at(self, timestamps):
        query = np.asarray(timestamps, dtype=np.int64).reshape(-1)
        if (np.any(query < self.times[0] - self.endpoint_tolerance_ns)
                or np.any(query > self.times[-1] + self.endpoint_tolerance_ns)):
            raise ValueError('POSE_COVERAGE_REJECTED: sensor timestamp outside selected trajectory')
        query = np.clip(query, self.times[0], self.times[-1])
        hi = np.minimum(np.searchsorted(self.times, query), len(self.times) - 1)
        lo = np.where(self.times[hi] == query, hi, np.maximum(hi - 1, 0))
        gap = self.times[hi] - self.times[lo]
        if np.any(gap > self.max_gap_ns):
            raise ValueError('POSE_COVERAGE_REJECTED: selected trajectory has a gap')
        alpha = np.divide(query - self.times[lo], gap, out=np.zeros(len(query)), where=gap != 0)[:, None]
        position = self.positions[lo] * (1 - alpha) + self.positions[hi] * alpha
        qa = self.quaternions[lo]
        qb = self.quaternions[hi].copy()
        qb[np.sum(qa * qb, axis=1) < 0] *= -1
        q = qa * (1 - alpha) + qb * alpha
        return position, Rotation.from_quat(q)

    def pose_at(self, timestamp):
        timestamp = int(timestamp)
        if timestamp not in self._pose_cache:
            position, rotation = self.poses_at([timestamp])
            self._pose_cache[timestamp] = {
                'position': dict(zip('xyz', position[0].tolist())),
                'orientation': dict(zip('xyzw', rotation.as_quat()[0].tolist()))}
        return {key: dict(value) for key, value in self._pose_cache[timestamp].items()}

    def deskew(self, xyz, point_timestamps, reference_timestamp):
        if np.any(np.abs(np.asarray(point_timestamps, np.int64) - int(reference_timestamp)) > 200_000_000):
            raise ValueError('Point timestamp is outside the reference scan')
        # Many LiDAR returns share a firing timestamp. Interpolate and normalize
        # each distinct quaternion once, retaining the original per-point order.
        times, inverse = np.unique(np.asarray(point_timestamps, np.int64), return_inverse=True)
        position, rotation = self.poses_at(times)
        position, rotation = position[inverse], rotation[inverse]
        origin, basis = self.poses_at([reference_timestamp])
        return basis.inv().apply(rotation.apply(xyz) + position - origin).astype(np.float32)

    @classmethod
    def load(cls, path):
        doc = json.loads(Path(path).read_text())
        return cls(doc['samples'], max_gap_ns=doc['max_gap_ns'],
                   endpoint_tolerance_ns=doc['endpoint_tolerance_ns'])


def load_geometry(clip_dir):
    path = Path(clip_dir) / 'geometry.json'
    return json.loads(path.read_text()) if path.is_file() else {}
