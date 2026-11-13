"""Derived-only refresh must retain every input and reuse unrelated products."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
import export_scene
import static_agg


class DerivedRefreshTests(unittest.TestCase):
    def test_shared_storage_reconstructs_full_points_and_identical_occ(self):
        pose = {'position': dict(x=0., y=0., z=0.), 'orientation': dict(x=0., y=0., z=0., w=1.)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / 'clip/frames/1'
            frame.mkdir(parents=True)
            pred = root / 'preds'
            pred.mkdir()
            cloud = np.array([[0, 5, 0, 0, 0, 0, 1], [1, 6, 1, 0, 0, 0, 2]], dtype=np.float32)
            cloud.tofile(frame / 'lidar_merge.bin')
            np.save(pred / '1_pred.npy', np.array([13, 0], dtype=np.int32))
            (frame / 'frame.json').write_text(json.dumps({'dependency': {'sensors': {}, 'ego_pose': {'pose': pose}}, 'timestamp': 1}))
            aggregate = {'xyz_map': np.array([[0, 5, 0], [1, 5, 0], [100, 5, 0]], np.float32),
                         'labels': np.array([13, 17, 13]), 'lidar_ids': np.array([1, 2, 1])}
            kwargs = dict(clip_id='clip', clip_dir=root / 'clip', ts='1', pred_dir=pred,
                          static_agg=aggregate, grid_size=.05, reuse_pred=True, x_range=(-10, 10),
                          y_range=(-10, 10), z_range=(-5, 5), occ_voxel=.25, occ_min_points=1,
                          export_points=True, max_export_points=0, ego_filter={'enabled': False},
                          pose_samples=[(0, pose), (2, pose)], geometry={})
            old, new = root / 'old', root / 'new'
            export_scene.export_frame(**kwargs, scene_root=old, out_frame=old / 'frames/1')
            export_scene.export_frame(**kwargs, scene_root=new, out_frame=new / 'frames/1',
                                      reuse_scene=old, shared_static=True)
            a = json.loads((old / 'frames/1/meta.json').read_text())
            b = json.loads((new / 'frames/1/meta.json').read_text())
            self.assertEqual(b['assets']['points']['n'], 1)
            self.assertEqual(b['assets']['points']['logical_n'], 3)
            for suffix in ('centers.f32', 'ijk.i32', 'counts.i32', 'labels.u8'):
                filename = f'frames/1/occ_{suffix}.bin'
                self.assertEqual((old / filename).read_bytes(), (new / filename).read_bytes())
            roi = b['assets']['points']['static_roi']
            xyz, labels, lidar = static_agg.static_in_vehicle(aggregate, np.eye(4),
                x_range=roi['x'], y_range=roi['y'], z_range=roi['z'])
            for key, part, dtype, width in [('xyz', xyz, np.float32, 3), ('labels', labels, np.uint8, 1),
                                          ('lidar_id', lidar, np.uint8, 1)]:
                expected = np.fromfile(old / a['assets']['points'][key]['uri'], dtype=dtype).reshape(-1, width)
                dynamic = np.fromfile(new / b['assets']['points'][key]['uri'], dtype=dtype).reshape(-1, width)
                np.testing.assert_array_equal(expected, np.concatenate([part.reshape(-1, width), dynamic]))
            self.assertEqual(a['assets']['cameras'], b['assets']['cameras'])
            self.assertEqual(a['assets']['frame_sensor_points'], b['assets']['frame_sensor_points'])

    def test_stride_sampling_and_point_caps_are_rejected(self):
        for args in (['--agg-stride', '5'], ['--static-voxel', '.25'], ['--max-export-points', '1'],
                     ['--reuse-scene', '/unused', '--reuse-pred'], ['--require-cached-preds']):
            with self.assertRaises(SystemExit):
                export_scene.main(['--clip', 'test', *args])

    def test_reused_assets_are_links_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, new = root / 'previous', root / 'new'
            previous.mkdir()
            old = previous / 'camera.jpg'
            old.write_bytes(b'original-camera')
            before = old.stat()
            group = [{'image': {'uri': 'camera.jpg', 'shape': [1]}}]
            export_scene.link_reused_assets(new, previous, group)
            export_scene.link_reused_assets(new, previous, group)
            self.assertTrue((new / 'camera.jpg').is_symlink())
            self.assertEqual((new / 'camera.jpg').read_bytes(), b'original-camera')
            self.assertEqual(old.stat().st_mtime_ns, before.st_mtime_ns)
            for uri in ('../outside', str(old)):
                with self.assertRaises(ValueError):
                    export_scene.link_reused_assets(new, previous, {'uri': uri})

    def test_export_selection_does_not_sample_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'cache/test'
            clip.mkdir(parents=True)
            timestamps = ['1000000000', '2000000000', '3000000000']
            (clip / 'frames_index.json').write_text(json.dumps([
                {'timestamp': ts, 'has_lidar': True} for ts in timestamps]))
            seen = {}
            def capture(**kwargs):
                seen.update(kwargs)
                raise RuntimeError('captured before generation')
            with patch('pose_trajectory.load_geometry', return_value={}), \
                    patch.object(static_agg, 'load_pose_samples', return_value=[(0, {}), (4_000_000_000, {})]), \
                    patch.object(export_scene.qgate, 'assess_clip_geometry', return_value={'allow_occ': True}), \
                    patch.object(export_scene, 'ensure_preds', side_effect=capture):
                result = export_scene.main([
                    '--clip', 'test', '--backup-root', str(root / 'cache'), '--out-dir', str(root / 'scenes'),
                    '--pred-dir', str(root / 'preds'), '--reuse-pred', '--stride', '2', '--max-frames', '1',
                    '--skip-initial-seconds', '1'])
            self.assertEqual(result, 1)
            self.assertEqual(seen['timestamps'], ['2000000000'])
            self.assertEqual(seen['agg_timestamps'], timestamps)

    def test_missing_static_input_fails_instead_of_skipping_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                static_agg.load_or_build_static_aggregate(
                    root, root, ['1'], load_lidar_bin=lambda *a, **kw: None, lidar_cols=7,
                    pose_samples=[(0, {}), (2, {})])


if __name__ == '__main__':
    unittest.main()
