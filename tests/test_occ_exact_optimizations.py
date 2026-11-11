import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/occ'))
import pipeline
import static_agg as sag
import occupancy
from pose_trajectory import PoseSamples


class ExactOptimizationTests(unittest.TestCase):
    def test_fused_packing_matches_numpy_at_voxel_and_roi_boundaries(self):
        rng = np.random.default_rng(31)
        for voxel in (.2, .3, .07):
            bounds = ((-30.1, 30.2), (-20.3, 40.4), (-5.1, 20.2))
            xyz = rng.uniform([-32, -22, -7], [32, 42, 22], (15000, 3)).astype(np.float32)
            edges = np.array([lo + np.arange(100, dtype=np.float32) * np.float32(voxel)
                              for lo, _ in bounds], dtype=np.float32).T
            xyz = np.concatenate([xyz, edges, np.nextafter(edges, np.float32(-np.inf)),
                                  np.nextafter(edges, np.float32(np.inf))])
            labels = rng.integers(0, 22, len(xyz), dtype=np.int32)
            for minimum in (1, 2):
                kwargs = dict(x_range=bounds[0], y_range=bounds[1], z_range=bounds[2],
                              voxel=voxel, min_points=minimum)
                with patch('voxel_votes.pack_voxels', None):
                    expected = occupancy.build_occupancy(xyz, labels, **kwargs)
                actual = occupancy.build_occupancy(xyz, labels, **kwargs)
                for name in ('ijk', 'centers', 'labels', 'counts', 'max_z'):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_duplicate_firing_times_preserve_every_point_bit(self):
        times=np.arange(10,dtype=np.int64)*100_000_000+1_700_000_000_000_000_000
        rotations=Rotation.from_euler('xyz',np.arange(30).reshape(10,3)/100).as_quat()
        samples=[(int(t),{'position':dict(zip('xyz',[i*.3,i*.4,i*.05])),
                            'orientation':dict(zip('xyzw',rotations[i]))}) for i,t in enumerate(times)]
        trajectory=PoseSamples(samples)
        rng=np.random.default_rng(12)
        queries=rng.choice(times[3]+np.arange(1000)*100000,size=10000)
        xyz=rng.normal(size=(len(queries),3)).astype(np.float32)*100
        reference=times[4]
        position,rotation=trajectory.poses_at(queries)
        origin,basis=trajectory.poses_at([reference])
        expected=basis.inv().apply(rotation.apply(xyz)+position-origin).astype(np.float32)
        np.testing.assert_array_equal(trajectory.deskew(xyz,queries,reference),expected)
        pose=trajectory.pose_at(reference);pose['position']['x']=999
        self.assertNotEqual(trajectory.pose_at(reference)['position']['x'],999)
        matrix=sag.interpolate_pose_matrix(trajectory,reference);matrix[:]=999
        np.testing.assert_array_equal(sag.interpolate_pose_matrix(trajectory,reference),
                                      sag.ego_pose_to_T_map_vehicle(trajectory.pose_at(reference)))

    def test_cached_map_transform_preserves_original_order_and_bits(self):
        rng=np.random.default_rng(11)
        xyz=rng.normal(size=(10003,3)).astype(np.float32)*100
        agg={'xyz_map':xyz,'labels':rng.integers(0,22,len(xyz)),'lidar_ids':np.ones(len(xyz))}
        for angle in [0,.1,1.8,-2.5]:
            matrix=np.eye(4);matrix[:3,:3]=Rotation.from_euler('z',angle).as_matrix();matrix[:3,3]=[3.,1.,2.]
            expected=sag.transform_points(xyz,np.linalg.inv(matrix))
            keep=(expected[:,0]>=-40)&(expected[:,0]<=40)
            actual,labels,lid=sag.static_in_vehicle(agg,matrix,x_range=(-40,40))
            np.testing.assert_array_equal(actual,expected[keep])
            np.testing.assert_array_equal(labels,agg['labels'][keep])
        agg['xyz_map']=xyz.copy()+1
        actual,*_=sag.static_in_vehicle(agg,np.eye(4))
        np.testing.assert_array_equal(actual,agg['xyz_map'])

    def test_materialization_reuses_snapshot_and_only_selected_cloud(self):
        frame={'timestamp':1,'clip_id':'clip','dependency':{'sensors':{
            'lidar_merge_deskew':{'md5':'a'*32},'lidar_merge_raw':{'md5':'b'*32}}}}
        clip={'clip_id':'clip'}
        db=MagicMock();db.__getitem__.side_effect=AssertionError('Source was fetched twice')
        selected=[]
        def write(raw,path):
            selected.append(raw['dependency']['sensors']['lidar_merge_raw']['md5'])
            path.parent.mkdir(parents=True,exist_ok=True);np.zeros((3,7),np.float32).tofile(path)
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.materialize_raw_cache(db,'frames','clips','clip',Path(tmp),raw_clip=clip,
                                           frames=[frame],materialize_lidar=write)
            index=json.loads((Path(tmp)/'frames_index.json').read_text())
            self.assertEqual(index[0]['md5'],'a'*32)
            self.assertEqual(selected,['b'*32])
            self.assertEqual(json.loads((Path(tmp)/'frames/1/frame.json').read_text()),frame)


if __name__ == '__main__':unittest.main()
