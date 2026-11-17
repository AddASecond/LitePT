"""Content-addressed S3 blobs, exposed through the consumer's existing bucket mount."""
import base64
import hashlib
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor

MULTIPART_THRESHOLD = 64 * 1024**2
MULTIPART_PART_SIZE = 16 * 1024**2


def s3_publisher(bucket, prefix, mount_root, *, profile=None, endpoint_url=None, client=None, checksums=None):
    if not bucket or not Path(mount_root).is_absolute():
        raise ValueError('S3 bucket and absolute consumer mount root are required')
    prefix = prefix.strip('/')
    if '..' in PurePosixPath(prefix).parts:
        raise ValueError('invalid S3 asset prefix')
    if client is None:
        import boto3
        from botocore.config import Config
        client = boto3.Session(profile_name=profile).client('s3', endpoint_url=endpoint_url, config=Config(
            connect_timeout=10, read_timeout=120, max_pool_connections=32,
            retries={'mode': 'standard', 'max_attempts': 5}))
    from botocore.exceptions import ClientError
    uploaded = {}

    def verify(key, head, length, digest, md5):
        remote_sha = head.get('Metadata', {}).get('sha256')
        if head['ContentLength'] != length or remote_sha not in (None, digest):
            raise ValueError(f'S3 digest/length conflict: {bucket}/{key}')
        # Some gateways strip user metadata. Single-part SSE-S3 ETags are
        # content MD5s; other ETags require reading the object to verify it.
        if remote_sha is None and head.get('ETag', '').strip('"') != md5.hex():
            remote_digest = hashlib.sha256()
            with client.get_object(Bucket=bucket, Key=key)['Body'] as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    remote_digest.update(chunk)
            if remote_digest.hexdigest() != digest:
                raise ValueError(f'S3 digest/length conflict: {bucket}/{key}')

    def existing(key):
        try:
            return client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response['Error']['Code'] not in ('404', 'NoSuchKey', 'NotFound'):
                raise
            return None

    def publish(path, *, source_camera_md5=None, data=None):
        path = Path(path)
        stat = path.stat() if data is None else None
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) if stat else None
        known = (checksums or {}).get(signature) if data is None else None
        if known is not None:
            digest, md5 = known
            length = stat.st_size
        else:
            data = path.read_bytes() if data is None else data
            digest = hashlib.sha256(data).hexdigest()
            md5 = hashlib.md5(data).digest()
            length = len(data)
        suffix = ''.join(Path(path).suffixes) or '.bin'
        key = str(PurePosixPath(prefix) / digest[:2] / digest[2:4] / (digest[4:] + suffix))
        head = None
        if source_camera_md5 is not None:
            if md5.hex() != source_camera_md5.lower():
                raise ValueError(f'Source camera MD5 mismatch: {path}')
            raw_key = f'rawdata/camera/{md5.hex()[:2]}/{md5.hex()[2:4]}/{md5.hex()[4:]}.jpg'
            if raw_key in uploaded:
                return dict(uploaded[raw_key])
            head = existing(raw_key)
            if head is not None:
                key = raw_key
        if key in uploaded:
            return dict(uploaded[key])
        if head is None:
            head = existing(key)
        if head is None:
            if data is None:
                data = path.read_bytes()
            if length >= MULTIPART_THRESHOLD:
                upload_id = client.create_multipart_upload(Bucket=bucket, Key=key,
                    Metadata={'sha256': digest}, ContentType='application/octet-stream')['UploadId']
                try:
                    def send_part(index):
                        body = data[index*MULTIPART_PART_SIZE:(index+1)*MULTIPART_PART_SIZE]
                        uploaded_part = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                            PartNumber=index+1, Body=body,
                            ContentMD5=base64.b64encode(hashlib.md5(body).digest()).decode())
                        return {'PartNumber': index+1, 'ETag': uploaded_part['ETag']}
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        parts = list(pool.map(send_part, range((length+MULTIPART_PART_SIZE-1)//MULTIPART_PART_SIZE)))
                    response = client.complete_multipart_upload(Bucket=bucket, Key=key,
                        UploadId=upload_id, MultipartUpload={'Parts': parts})
                except BaseException:
                    try:
                        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
                    except Exception:
                        pass
                    raise
            else:
                response = client.put_object(Bucket=bucket, Key=key, Body=data,
                              ContentMD5=base64.b64encode(md5).decode(),
                              Metadata={'sha256': digest},
                              ContentType='image/jpeg' if suffix.lower() in ('.jpg', '.jpeg') else 'application/octet-stream')
            # The successful PUT validates Content-MD5 on the server. Its
            # matching ETag confirms the stored bytes without another HEAD.
            if response.get('ETag', '').strip('"') != md5.hex():
                head = client.head_object(Bucket=bucket, Key=key)
        if head is not None:
            verify(key, head, length, digest, md5)
        result = {'storage': 's3', 'bucket': bucket, 'key': key,
                  'uri': str(Path(mount_root) / key), 'sha256': digest, 'length': length}
        uploaded[key] = result
        return dict(result)

    def reference_from_checksum(path, digest, length):
        suffix = ''.join(Path(path).suffixes) or '.bin'
        key = str(PurePosixPath(prefix) / digest[:2] / digest[2:4] / (digest[4:] + suffix))
        return {'storage': 's3', 'bucket': bucket, 'key': key,
                'uri': str(Path(mount_root) / key), 'sha256': digest, 'length': length}
    publish.reference_from_checksum = reference_from_checksum
    return publish
