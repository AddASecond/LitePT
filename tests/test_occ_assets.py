"""Regression coverage for the Mongo -> Scene -> asset publication contract."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
import export_scene
import store


class CameraContractTests(unittest.TestCase):
    def test_modern_names_prefer_matching_undistort_and_include_camera12(self):
        sensors = {name: {} for name in ('camera1_raw', 'camera1_undistort', 'camera1',
                                        'camera12_undistort', 'camera2', 'lidar_merge_deskew')}
        self.assertEqual(export_scene.camera_names(sensors),
                         ['camera1_undistort', 'camera2', 'camera12_undistort'])

    def test_intrinsics_are_taken_from_selected_variant(self):
        matrix = [[500, 0, 512], [0, 510, 288], [0, 0, 1]]
        camera = {'intrinsic': {'intrinsic': matrix, 'distortion': [[0] * 8],
                                'width': 1024, 'height': 576},
                  'extrinsic': {'transformation': np.eye(4).tolist()}}
        k, distortion, *_ = export_scene.parse_camera(camera)
        np.testing.assert_array_equal(k, matrix)
        np.testing.assert_array_equal(distortion, np.zeros(5))
        camera['intrinsic']['distortion'][0][5] = 0.2
        with self.assertRaisesRegex(ValueError, 'rational distortion'):
            export_scene.parse_camera(camera)

    def test_old_scene_without_calibration_is_not_publishable(self):
        with self.assertRaisesRegex(ValueError, 'no calibrated cameras'):
            store.validate_camera_assets({'assets': {'cameras': []}}, [{'md5': 'a' * 32}])
        with self.assertRaisesRegex(ValueError, 'incomplete camera calibration'):
            store.validate_camera_assets({'assets': {'cameras': [{'name': 'camera1', 'uri': 'image.jpg'}]}}, [])


class PublicationTests(unittest.TestCase):
    def test_missing_sensor_blob_and_wrong_shape_fail_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            asset = {'uri': 'sensor.bin', 'dtype': 'float32', 'shape': [2, 3]}
            with self.assertRaises(FileNotFoundError):
                store.validate_scene_assets(scene, {'frame_sensor_points': {'xyz': asset}})
            (scene / 'sensor.bin').write_bytes(b'\0' * 12)
            with self.assertRaisesRegex(ValueError, 'shape/length mismatch'):
                store.validate_scene_assets(scene, {'frame_sensor_points': {'xyz': asset}})

    def test_gridfs_group_preserves_array_contract_without_local_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            (scene / 'points.bin').write_bytes(b'\0' * 24)
            group = {'n': 2, 'source': 'lidar_merge_deskew',
                     'xyz': {'uri': 'points.bin', 'dtype': 'float32', 'shape': [2, 3]}}
            published = store.upload_group(scene, group, scene, publish=lambda p: {
                'storage': 'gridfs', 'bucket': 'occ_blobs', 'gridfs_id': 'a' * 24, 'length': p.stat().st_size})
            self.assertEqual(published['n'], 2)
            self.assertEqual(published['source'], 'lidar_merge_deskew')
            self.assertEqual(published['xyz']['shape'], [2, 3])
            self.assertEqual(published['xyz']['length'], 24)
            self.assertNotIn('uri', published['xyz'])

    def test_gridfs_reuses_hash_and_rejects_wrong_existing_length(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / 'array.bin'
            asset.write_bytes(b'1234')
            db = MagicMock()
            db.__getitem__.return_value.find_one.return_value = {'_id': 'a' * 24, 'length': 4}
            with patch('gridfs.GridFSBucket') as bucket:
                publish = store.gridfs_publisher(db, 'occ_blobs')
                result = publish(asset)
                self.assertEqual(result['gridfs_id'], 'a' * 24)
                bucket.return_value.upload_from_stream.assert_not_called()
                db.__getitem__.return_value.find_one.return_value['length'] = 5
                with self.assertRaisesRegex(ValueError, 'digest/length conflict'):
                    publish(asset)


if __name__ == '__main__':
    unittest.main()
