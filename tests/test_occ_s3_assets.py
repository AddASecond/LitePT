import base64
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/occ'))
from s3_assets import s3_publisher


class S3AssetTests(unittest.TestCase):
    def test_new_blob_has_server_checksum_and_consumer_file_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'occ.i32.bin'
            path.write_bytes(b'1234')
            digest = hashlib.sha256(b'1234').hexdigest()
            client = MagicMock()
            client.head_object.side_effect = ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            client.put_object.return_value = {'ETag': '"' + hashlib.md5(b'1234').hexdigest() + '"'}
            publish = s3_publisher('data', 'occ', '/mnt/data', client=client)
            result = publish(path)
            self.assertEqual(result['uri'], f'/mnt/data/occ/{digest[:2]}/{digest[2:4]}/{digest[4:]}.i32.bin')
            self.assertEqual(result['length'], 4)
            self.assertEqual(client.put_object.call_args.kwargs['ContentMD5'],
                             base64.b64encode(hashlib.md5(b'1234').digest()).decode())
            self.assertEqual(publish(path), result)
            self.assertEqual(client.put_object.call_count, 1)
            self.assertEqual(client.head_object.call_count, 1)

    def test_put_without_md5_etag_still_verifies_remote_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'array.bin'
            path.write_bytes(b'1234')
            for remote, succeeds in [(b'1234', True), (b'bad!', False)]:
                client = MagicMock()
                client.put_object.return_value = {'ETag': 'opaque'}
                client.head_object.side_effect = [ClientError({'Error': {'Code': '404'}}, 'HeadObject'),
                                                  {'ContentLength': 4, 'ETag': 'opaque'}]
                client.get_object.return_value = {'Body': io.BytesIO(remote)}
                publish = s3_publisher('data', 'occ', '/mnt/data', client=client)
                if succeeds:
                    publish(path)
                else:
                    with self.assertRaisesRegex(ValueError, 'digest/length conflict'):
                        publish(path)
                self.assertEqual(client.head_object.call_count, 2)

    def test_source_camera_reuses_only_verified_original_and_falls_back_if_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera1_undistort.jpg'
            path.write_bytes(b'image')
            md5 = hashlib.md5(b'image').hexdigest()
            client = MagicMock()
            client.head_object.return_value = {'ContentLength': 5, 'ETag': f'"{md5}"'}
            publish = s3_publisher('data', 'occ', '/mnt/data', client=client)
            result = publish(path, source_camera_md5=md5)
            self.assertEqual(result['key'], f'rawdata/camera/{md5[:2]}/{md5[2:4]}/{md5[4:]}.jpg')
            self.assertEqual(result['sha256'], hashlib.sha256(b'image').hexdigest())
            client.put_object.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'Source camera MD5 mismatch'):
                publish(path, source_camera_md5='0' * 32)
            client.head_object.side_effect = ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            client.put_object.return_value = {'ETag': f'"{md5}"'}
            result = s3_publisher('data', 'occ', '/mnt/data', client=client)(path, source_camera_md5=md5)
            self.assertTrue(result['key'].startswith('occ/'))
            self.assertTrue(client.put_object.call_args.kwargs['Key'].startswith('occ/'))

    def test_corrupt_source_camera_is_not_published_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera.jpg'
            path.write_bytes(b'image')
            client = MagicMock()
            client.head_object.return_value = {'ContentLength': 3}
            with self.assertRaisesRegex(ValueError, 'digest/length conflict'):
                s3_publisher('data', 'occ', '/mnt/data', client=client)(
                    path, source_camera_md5=hashlib.md5(b'image').hexdigest())
            client.put_object.assert_not_called()

    def test_remote_conflict_and_access_denial_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.bin'
            path.write_bytes(b'1234')
            client = MagicMock()
            client.head_object.return_value = {'ContentLength': 4, 'Metadata': {'sha256': 'wrong'}}
            publish = s3_publisher('data', 'occ', '/mnt/data', client=client)
            with self.assertRaisesRegex(ValueError, 'digest/length conflict'):
                publish(path)
            client.head_object.side_effect = ClientError({'Error': {'Code': '403'}}, 'HeadObject')
            with self.assertRaises(ClientError):
                publish(path)
            client.put_object.assert_not_called()

    def test_unsafe_prefix_or_missing_mount_is_rejected(self):
        for prefix, root in [('../escape', '/mnt/data'), ('occ', '')]:
            with self.assertRaises(ValueError):
                s3_publisher('data', prefix, root, client=MagicMock())

    def test_gateway_without_user_metadata_uses_etag_or_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'array.bin'
            path.write_bytes(b'1234')
            client = MagicMock()
            client.head_object.return_value = {'ContentLength': 4, 'Metadata': {},
                                               'ETag': '"' + hashlib.md5(b'1234').hexdigest() + '"'}
            s3_publisher('data', 'occ', '/mnt/data', client=client)(path)
            client.get_object.assert_not_called()
            client.head_object.return_value['ETag'] = 'multipart-2'
            client.get_object.return_value = {'Body': io.BytesIO(b'1234')}
            s3_publisher('data', 'occ', '/mnt/data', client=client)(path)
            client.get_object.return_value = {'Body': io.BytesIO(b'bad!')}
            with self.assertRaisesRegex(ValueError, 'digest/length conflict'):
                s3_publisher('data', 'occ', '/mnt/data', client=client)(path)


if __name__ == '__main__':
    unittest.main()
