import base64
import hashlib
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
            client.head_object.side_effect = [ClientError({'Error': {'Code': '404'}}, 'HeadObject'),
                                              {'ContentLength': 4, 'Metadata': {'sha256': digest}}]
            publish = s3_publisher('data', 'occ', '/mnt/data', client=client)
            result = publish(path)
            self.assertEqual(result['uri'], f'/mnt/data/occ/{digest[:2]}/{digest[2:4]}/{digest[4:]}.i32.bin')
            self.assertEqual(result['length'], 4)
            self.assertEqual(client.put_object.call_args.kwargs['ContentMD5'],
                             base64.b64encode(hashlib.md5(b'1234').digest()).decode())
            self.assertEqual(publish(path), result)
            self.assertEqual(client.put_object.call_count, 1)

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


if __name__ == '__main__':
    unittest.main()
