from botocore.exceptions import ClientError

from app.common.retry_util import with_retry
from app.common.s3.client import (
    get_s3_client,
    get_s3_object,
    head_s3_object,
    put_s3_object,
    upload_s3_object,
)
from app.common.s3.path import split_s3_path


@with_retry
def _get_s3_object_or_ex(path: str, client, **kwargs):
    if not client:
        client = get_s3_client(path)
    try:
        return get_s3_object(client, path, **kwargs)
    except ClientError as e:
        return e


def get_s3_object_with_retry(path: str, client=None, **kwargs):
    ret = _get_s3_object_or_ex(path, client, **kwargs)
    if isinstance(ret, ClientError):
        raise ret
    return ret


@with_retry
def _head_s3_object_or_ex(path: str, raise_404: bool, client):
    if not client:
        client = get_s3_client(path)
    try:
        return head_s3_object(client, path, raise_404)
    except ClientError as e:
        return e


def head_s3_object_with_retry(path: str, raise_404=False, client=None):
    ret = _head_s3_object_or_ex(path, raise_404, client)
    if isinstance(ret, ClientError):
        raise ret
    return ret


@with_retry(sleep_time=6)
def put_s3_object_with_retry(path: str, body: bytes, client=None):
    if not client:
        client = get_s3_client(path)
    put_s3_object(client, path, body)


@with_retry(sleep_time=180)
def s3_upload_with_retry(path: str, local_file_path: str, client=None):
    if not client:
        client = get_s3_client(path)
    upload_s3_object(client, path, local_file_path)


@with_retry(sleep_time=180)
def upload_s3_object_with_retry(path: str, local_file_path: str, client=None):
    if not client:
        client = get_s3_client(path)
    upload_s3_object(client, path, local_file_path)


@with_retry(sleep_time=180)
def download_s3_file_with_retry(path: str, local_file_path: str, client=None):
    if not client:
        client = get_s3_client(path)
    bucket, key = split_s3_path(path)
    client.download_file(bucket, key, local_file_path)
