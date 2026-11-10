"""Content-addressed S3 blobs, exposed through the consumer's existing bucket mount."""
import base64
import hashlib
from pathlib import Path, PurePosixPath


def s3_publisher(bucket, prefix, mount_root, *, profile=None, client=None):
    if not bucket or not Path(mount_root).is_absolute():
        raise ValueError('S3 bucket and absolute consumer mount root are required')
    prefix = prefix.strip('/')
    if '..' in PurePosixPath(prefix).parts:
        raise ValueError('invalid S3 asset prefix')
    if client is None:
        import boto3
        from botocore.config import Config
        client = boto3.Session(profile_name=profile).client('s3', config=Config(
            connect_timeout=10, read_timeout=120, retries={'mode': 'standard', 'max_attempts': 5}))
    from botocore.exceptions import ClientError
    uploaded = {}

    def publish(path):
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        suffix = ''.join(Path(path).suffixes) or '.bin'
        key = str(PurePosixPath(prefix) / digest[:2] / digest[2:4] / (digest[4:] + suffix))
        if key in uploaded:
            return dict(uploaded[key])
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response['Error']['Code'] not in ('404', 'NoSuchKey', 'NotFound'):
                raise
            client.put_object(Bucket=bucket, Key=key, Body=data,
                              ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode(),
                              Metadata={'sha256': digest},
                              ContentType='image/jpeg' if suffix.lower() in ('.jpg', '.jpeg') else 'application/octet-stream')
            head = client.head_object(Bucket=bucket, Key=key)
        if head['ContentLength'] != len(data) or head.get('Metadata', {}).get('sha256') != digest:
            raise ValueError(f'S3 digest/length conflict: {bucket}/{key}')
        result = {'storage': 's3', 'bucket': bucket, 'key': key,
                  'uri': str(Path(mount_root) / key), 'sha256': digest, 'length': len(data)}
        uploaded[key] = result
        return dict(result)

    return publish
