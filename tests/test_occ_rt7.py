import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
import occupancy
import rt7


class RT7Tests(unittest.TestCase):
    def test_mapping_and_invalid_inputs(self):
        np.testing.assert_array_equal(rt7.map_labels(np.arange(22)),
            [2,2,2,2,2,2,2,6,6,6,6,2,2,5,4,4,3,0,0,1,1,1])
        self.assertEqual(len(rt7.NAMES), 7)
        for values in ([-1], [22], [1.5]):
            with self.assertRaises(ValueError):
                rt7.map_labels(values)

    def test_merge_classes_before_voxel_voting(self):
        # Fine-class voting picks Building (3), but merged Drivable has 4 votes.
        labels = np.array([13,13,13,17,17,18,18])
        cloud = np.tile([0.1,0.1,0.1], (len(labels), 1)).astype(np.float32)
        grid = occupancy.build_occupancy(cloud, rt7.map_labels(labels),
            x_range=(0,1), y_range=(0,1), z_range=(0,1), voxel=.25)
        np.testing.assert_array_equal(grid.labels, [0])
        np.testing.assert_array_equal(grid.counts, [7])

    def test_ties_choose_lowest_rt7_id(self):
        labels = np.array([0,17])  # Movable, Drivable
        grid = occupancy.build_occupancy(np.full((2,3), .1, np.float32), rt7.map_labels(labels),
            x_range=(0,1), y_range=(0,1), z_range=(0,1), voxel=.25)
        np.testing.assert_array_equal(grid.labels, [0])


if __name__ == '__main__':
    unittest.main()
