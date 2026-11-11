import sys
import types
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
import occupancy
import pipeline


class EfficiencyTests(unittest.TestCase):
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
