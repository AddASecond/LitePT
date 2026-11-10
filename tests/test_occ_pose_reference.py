import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
import static_agg as sag
import store


def pose(x):
    return {'position': {'x': x, 'y': 0., 'z': 0.},
            'orientation': {'x': 0., 'y': 0., 'z': 0., 'w': 1.}}


class PoseReferenceTests(unittest.TestCase):
    def test_map_camera_chain_cancels_only_matching_lidar_reference(self):
        samples = [(0, pose(0)), (100_000_000, pose(10))]
        lidar = sag.interpolate_pose_matrix(samples, 25_000_000)
        camera = sag.interpolate_pose_matrix(samples, 75_000_000)
        nearest = sag.ego_pose_to_T_map_vehicle(pose(0))
        extrinsic = np.eye(4)
        extrinsic[:3,3] = [1., 2., 3.]
        compensation = extrinsic @ np.linalg.inv(camera) @ lidar
        direct = extrinsic @ np.linalg.inv(camera)
        np.testing.assert_allclose(compensation @ np.linalg.inv(lidar), direct)
        self.assertGreater(np.max(np.abs(compensation @ np.linalg.inv(nearest) - direct)), 1)

    def test_publication_uses_scene_reference_pose_and_exact_nanoseconds(self):
        ts = 1772694962198784000
        raw = {'timestamp': ts, 'dependency': {'ego_pose': {'pose': pose(10)}}}
        scene = {'pose_reference': 'lidar_timestamp', 'ego_pose_timestamp_ns': str(ts), 'ego_pose': pose(9.8)}
        published = store.published_ego_pose(scene, raw)
        self.assertEqual(published['pose'], pose(9.8))
        self.assertEqual(published['header']['stamp'], {'sec': 1772694962, 'nanosec': 198784000})
        with self.assertRaisesRegex(ValueError, 're-export'):
            store.published_ego_pose({'ego_pose': pose(10)}, raw)

    def test_aggregate_uses_interpolated_pose_and_invalidates_neighbor_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / 'frames/25000000'
            frame.mkdir(parents=True)
            pred = root / 'preds'; pred.mkdir()
            points = np.array([[3., 6., 0., 1., 0., 0., 1.]], dtype=np.float32)
            points.tofile(frame / 'lidar_merge.bin')
            np.save(pred / '25000000_pred.npy', np.array([13], dtype=np.int32))
            (frame / 'frame.json').write_text(json.dumps({'timestamp': 25_000_000, 'dependency': {
                'ego_pose': {'pose': pose(10)}, 'sensors': {'lidar_merge_deskew': {'md5': 'a'*32}}}}))
            kwargs = dict(clip_dir=root, pred_dir=pred, timestamps=['25000000'],
                          load_lidar_bin=lambda p, num_cols: np.fromfile(p, np.float32).reshape(-1, num_cols),
                          lidar_cols=7, voxel=.01, cache_path=root / 'aggregate.npz', use_oracle_boxes=False)
            samples = [(0, pose(0)), (100_000_000, pose(10))]
            first = sag.load_or_build_static_aggregate(**kwargs, pose_samples=samples)
            np.testing.assert_allclose(first['xyz_map'], [[5.5, 6., 0.]])
            self.assertTrue(sag.load_or_build_static_aggregate(**kwargs, pose_samples=samples)['from_cache'])
            second = sag.load_or_build_static_aggregate(**kwargs, pose_samples=[(0, pose(0)), (100_000_000, pose(20))])
            self.assertFalse(second['from_cache'])
            np.testing.assert_allclose(second['xyz_map'], [[8., 6., 0.]])


if __name__ == '__main__':
    unittest.main()
