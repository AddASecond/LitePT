import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
from camera_calibration import camera_metadata
from pose_trajectory import PoseSamples
from pipeline import pcd_binary_to_litept
import static_agg as sag
import quality_gate


def pose(position, angle=0):
    return {'position': dict(zip('xyz', position)),
            'orientation': dict(zip('xyzw', Rotation.from_euler('z', angle).as_quat()))}


class SelectedGeometryTests(unittest.TestCase):
    def test_geometry_gate_accepts_selected_pose_without_original_ego_pose(self):
        points = np.array([[1, 10, 0, 0, 0, 0, 1]], dtype=np.float32)
        transform = np.eye(4)
        transform[0, 3] = 2
        actual = quality_gate._structural_map(points, {}, .1, 100, pose_matrix=transform)
        np.testing.assert_allclose(actual, [[3, 10, 0]])

    def test_camera_basis_conversion_precedes_camera_time_compensation(self):
        samples = PoseSamples([(0, pose([0, 0, 0])), (100_000_000, pose([2, 3, 0], .2))])
        bridge = np.eye(4)
        bridge[:3, :3] = Rotation.from_euler('y', .3).as_matrix()
        bridge[:3, 3] = [.4, .5, .6]
        cam = {'timestamp': 80_000_000, 'extrinsic': {'transformation': np.eye(4).tolist()},
               'intrinsic': {'width': 10, 'height': 10, 'intrinsic': np.eye(3).tolist(), 'distortion': [0]*8}}
        geometry = {'calibration': {'T_camera_basis_point_basis': bridge.reshape(-1).tolist()},
                    'pose_source': {'decision': 'use_lio'}}
        result = camera_metadata('camera1_raw', cam, (10, 10), 'image.jpg', 20_000_000, samples, geometry)
        ref = sag.interpolate_pose_matrix(samples, 20_000_000)
        capture = sag.interpolate_pose_matrix(samples, 80_000_000)
        actual = np.array(result['T_c_v_lidar_ref']).reshape(4, 4)
        np.testing.assert_allclose(actual @ np.linalg.inv(ref), bridge @ np.linalg.inv(capture), atol=1e-12)
        self.assertGreater(np.max(abs(actual - np.linalg.inv(capture) @ ref @ bridge)), .1)
        self.assertEqual(result['time_compensation']['pose_source'], 'use_lio')

    def test_deskew_and_camera_observe_same_stationary_world_point(self):
        trajectory = PoseSamples([(0, pose([0, 0, 0])), (100_000_000, pose([2, 3, 0], .2))])
        times = np.array([20_000_000, 40_000_000, 60_000_000])
        position, rotation = trajectory.poses_at(times)
        world = np.tile([10., 20., 5.], (3, 1))
        raw = rotation.inv().apply(world - position)
        fixed = trajectory.deskew(raw, times, 50_000_000)
        ref_pos, ref_rot = trajectory.poses_at([50_000_000])
        np.testing.assert_allclose(fixed, ref_rot.inv().apply(world - ref_pos), atol=2e-6)
        np.testing.assert_allclose(fixed[0], fixed[-1], atol=2e-6)

    def test_epoch_nanoseconds_and_missing_coverage(self):
        epoch = 1774493749301238016
        trajectory = PoseSamples([(epoch, pose([0, 0, 0])), (epoch + 100_000_000, pose([1, 0, 0]))])
        self.assertAlmostEqual(trajectory.pose_at(epoch + 1_000_001)['position']['x'], .01000001, places=12)
        with self.assertRaisesRegex(ValueError, 'POSE_COVERAGE_REJECTED'):
            trajectory.pose_at(epoch - 20_000_000)
        gapped = PoseSamples([(0, pose([0, 0, 0])), (1_000_000_000, pose([1, 0, 0]))])
        with self.assertRaisesRegex(ValueError, 'trajectory has a gap'):
            gapped.pose_at(500_000_000)

    def test_lio_pcd_conversion_retains_point_times_until_deskew(self):
        epoch = 1774493749301238016
        trajectory = PoseSamples([(epoch, pose([0, 0, 0])), (epoch + 100_000_000, pose([1, 0, 0]))])
        data = np.array([(10., 0., 0., epoch, 1), (9., 0., 0., epoch+100_000_000, 2)],
                        dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('timestamp', '<i8'), ('lidar_id', 'u1')])
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'raw.pcd', Path(tmp)/'out.bin'
            source.write_bytes(b'FIELDS x y z timestamp lidar_id\nSIZE 4 4 4 8 1\nTYPE F F F I U\nCOUNT 1 1 1 1 1\nPOINTS 2\nDATA binary\n' + data.tobytes())
            pcd_binary_to_litept(source, target, trajectory=trajectory, reference_timestamp=epoch+50_000_000)
            actual = np.fromfile(target, np.float32).reshape(-1, 7)
            np.testing.assert_allclose(actual[:, :3], [[9.5, 0, 0], [9.5, 0, 0]])
            self.assertEqual(actual[:, 6].tolist(), [1, 2])


if __name__ == '__main__':
    unittest.main()
