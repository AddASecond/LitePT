import sys
import types
import unittest
import os
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
import occupancy
import pipeline
from paths import prediction_slot


class EfficiencyTests(unittest.TestCase):
    def test_prediction_slots_limit_models_and_release_after_failure(self):
        lock = threading.Lock()
        ready = threading.Barrier(2)
        release = threading.Event()
        third = threading.Event()
        active = 0
        peak = 0

        def worker(number):
            nonlocal active, peak
            with prediction_slot():
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    if number < 2:
                        ready.wait(timeout=5)
                        release.wait(timeout=5)
                    else:
                        third.set()
                    if number == 0:
                        raise ValueError('model failed')
                finally:
                    with lock:
                        active -= 1

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'OCC_PREDICTION_SLOTS': '2', 'OCC_PREDICTION_LOCK_DIR': tmp}):
            with ThreadPoolExecutor(max_workers=3) as pool:
                first = pool.submit(worker, 0)
                second = pool.submit(worker, 1)
                # Wait until both slots are actually occupied before submitting another.
                while ready.n_waiting != 0 or active < 2:
                    if first.done() or second.done():
                        self.fail('slot holders exited early')
                    threading.Event().wait(.001)
                last = pool.submit(worker, 2)
                self.assertFalse(third.wait(.05))
                release.set()
                with self.assertRaises(ValueError):
                    first.result(timeout=5)
                second.result(timeout=5)
                last.result(timeout=5)
        self.assertEqual(peak, 2)
        self.assertTrue(third.is_set())

    def test_exporter_leaves_cuda_initialization_to_prediction_process(self):
        fake_export = types.SimpleNamespace(main=lambda argv: 0)
        with patch.dict(sys.modules, {'export_scene': fake_export}), patch('paths.setup_cuda_env') as setup:
            self.assertEqual(pipeline._export_scene([]), 0)
        setup.assert_called_once_with(warmup=False)

    def test_voxel_votes_match_independent_counts_and_lowest_class_ties(self):
        rng = np.random.default_rng(8)
        xyz = rng.uniform(0, 3, (3000, 3)).astype(np.float32)
        labels = rng.integers(0, 22, len(xyz), dtype=np.int32)
        cells = defaultdict(list)
        for point, label in zip(xyz, labels):
            cells[tuple(np.floor(point/.5).astype(int))].append(int(label))
        for minimum in (1, 12):
            grid = occupancy.build_occupancy(xyz, labels, x_range=(0,3), y_range=(0,3),
                                            z_range=(0,3), voxel=.5, min_points=minimum)
            expected = {}
            for cell, values in cells.items():
                if len(values) >= minimum:
                    votes = Counter(values)
                    expected[cell] = (min(votes, key=lambda k: (-votes[k], k)), len(values))
            actual = {tuple(cell): (int(label), int(n)) for cell,label,n in zip(grid.ijk, grid.labels, grid.counts)}
            self.assertEqual(actual, expected)
        tie = occupancy.build_occupancy(np.full((4,3), .1), np.array([8,2,8,2]),
                                        x_range=(0,1), y_range=(0,1), z_range=(0,1), voxel=.5)
        self.assertEqual(tie.labels.tolist(), [2])


if __name__ == '__main__':
    unittest.main()
