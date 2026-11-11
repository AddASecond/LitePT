import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
from artifact_io import AssetPipeline
from s3_assets import s3_publisher


class StreamingTests(unittest.TestCase):
    def test_missing_clip_static_upload_prevents_all_mongo_writes(self):
        import store
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            static = scene / 'static.bin'
            static.write_bytes(b'1234')
            (scene / 'index.json').write_text(json.dumps({
                'frames': [{'timestamp': 1, 'meta_uri': 'meta.json'}],
                'static_agg': {'xyz': {'uri': 'static.bin', 'dtype': 'uint8', 'shape': [4]}}}))
            (scene / 'meta.json').write_text(json.dumps({'assets': {}}))
            raw = {'clip_id': 'clip', 'timestamp': 1, 'md5': 'a' * 32}
            sink = AssetPipeline(lambda *a, **k: {}, scene)
            client = MagicMock()
            db = client.__getitem__.return_value
            with patch.object(store, 'MongoClient', return_value=client), \
                    patch.object(store, 'md5_from_backup', return_value=raw['md5']), \
                    patch.object(store, 'resolve_raw', return_value=static), \
                    patch.object(store, 'raw_cameras', return_value=[]), \
                    patch.object(store, 'validate_camera_assets', return_value=[]), \
                    patch.object(store, 'published_ego_pose', return_value={}):
                with self.assertRaisesRegex(ValueError, 'Asset was not produced'):
                    store.run(scene=scene, raw_frame_collection='main_3dod_frames',
                              raw_clip_collection='main_3dod_clips',
                              occ_frame_collection='main_occ_frames', occ_clip_collection='main_occ_clips',
                              source_documents=({'clip_id': 'clip'}, [raw]),
                              asset_storage='s3', s3_bucket='data', s3_mount_root='/mnt/data',
                              prepublished=sink, write=True)
            db.__getitem__.return_value.update_one.assert_not_called()

    def test_backpressure_keeps_bytes_bounded_and_references_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = threading.Event()
            entered = threading.Event()
            def publish(path, *, data=None, source_camera_md5=None):
                entered.set()
                gate.wait(timeout=5)
                return {'sha256': hashlib.sha256(data).hexdigest()}
            sink = AssetPipeline(publish, root, max_bytes=8, workers=1)
            a, b = root/'a.bin', root/'b.bin'
            a.write_bytes(b'12345678');b.write_bytes(b'abcdefgh')
            sink.submit(a, data=a.read_bytes())
            self.assertTrue(entered.wait(5))
            accepted = threading.Event()
            def enqueue():
                sink.submit(b, data=b.read_bytes())
                accepted.set()
            thread = threading.Thread(target=enqueue);thread.start()
            self.assertFalse(accepted.wait(.05))
            gate.set();thread.join(5);sink.close()
            self.assertTrue(accepted.is_set())
            self.assertLessEqual(sink.peak_pending_bytes,8)
            self.assertEqual(sink.publish(a)['sha256'], hashlib.sha256(a.read_bytes()).hexdigest())
            a.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'changed'):
                sink.publish(a)

    def test_failed_upload_never_returns_a_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.bin';p.write_bytes(b'bytes')
            def fail(*args,**kwargs):raise ValueError('remote checksum conflict')
            sink=AssetPipeline(fail,Path(tmp));sink.submit(p,data=p.read_bytes())
            with self.assertRaisesRegex(RuntimeError,'publication failed'):sink.close()
            with self.assertRaisesRegex(ValueError,'remote checksum conflict'):sink.publish(p)

    def test_only_training_frames_are_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'frames/1/a.bin';p.parent.mkdir(parents=True);p.write_bytes(b'abc')
            sink=AssetPipeline(lambda *a,**k: self.fail('unselected frame uploaded'),root,timestamps=['2'])
            sink.submit(p,data=b'abc');sink.close();self.assertEqual(len(sink.futures),0)

    def test_verified_camera_reuses_hashes_without_reopening_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'camera.jpg';p.write_bytes(b'camera');data=p.read_bytes()
            md5=hashlib.md5(data).digest();sha=hashlib.sha256(data).hexdigest()
            cache={AssetPipeline.signature(p):(sha,md5)}
            client=MagicMock();client.head_object.return_value={'ContentLength':len(data),'ETag':md5.hex()}
            publisher=s3_publisher('data','occ','/mnt/data',client=client,checksums=cache)
            from unittest.mock import patch
            with patch.object(Path,'read_bytes',side_effect=AssertionError('image read twice')):
                result=publisher(p,source_camera_md5=md5.hex())
            self.assertEqual(result['sha256'],sha)
            self.assertTrue(result['key'].startswith('rawdata/camera/'))
            client.put_object.assert_not_called()
            p.write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'Source camera MD5 mismatch'):
                publisher(p,source_camera_md5=md5.hex())


if __name__ == '__main__':unittest.main()
