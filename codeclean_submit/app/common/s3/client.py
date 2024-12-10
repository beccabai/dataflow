import time
from typing import Dict, List, Union

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from botocore.exceptions import ClientError

from app.common.s3.conf import get_s3_config
from app.common.s3.path import split_s3_path


def get_s3_client(path: Union[str, List[str]], outside=False):
    s3_config = get_s3_config(path, outside)
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(
                s3={"addressing_style": "path"},
                retries={"max_attempts": 8, "mode": "standard"},
                connect_timeout=600,
                read_timeout=600,
            ),
        )
    except:
        # older boto3 do not support retries.mode param.
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8}),
        )


def is_s3_404_error(e: Exception):
    if not isinstance(e, ClientError):
        return False
    return (
        e.response.get("Error", {}).get("Code") in ["404", "NoSuchKey"]
        or e.response.get("Error", {}).get("Message") == "Not Found"
        or e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404
    )


def head_s3_object(client, path: str, raise_404=False) -> Union[Dict, None]:
    bucket, key = split_s3_path(path)
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        return resp
    except ClientError as e:
        if not raise_404 and is_s3_404_error(e):
            return None
        raise


def _restore_and_wait(client, bucket: str, key: str, path: str):
    while True:
        head = client.head_object(Bucket=bucket, Key=key)
        restore = head.get("Restore", "")
        if not restore:
            req = {"Days": 1, "GlacierJobParameters": {"Tier": "Standard"}}
            client.restore_object(Bucket=bucket, Key=key, RestoreRequest=req)
            print(f"restoration-started: {path}")
        elif 'ongoing-request="true"' in restore:
            print(f"restoration-ongoing: {path}")
        elif 'ongoing-request="false"' in restore:
            print(f"restoration-complete: {path}")
            break
        time.sleep(3)


def get_s3_object(client, path: str, **kwargs) -> dict:
    bucket, key = split_s3_path(path)
    try:
        return client.get_object(Bucket=bucket, Key=key, **kwargs)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "GlacierObjectNotRestore":
            _restore_and_wait(client, bucket, key, path)
            return client.get_object(Bucket=bucket, Key=key, **kwargs)
        raise


def put_s3_object(client, path: str, body: bytes):
    bucket, key = split_s3_path(path)
    return client.put_object(Bucket=bucket, Key=key, Body=body)


def upload_s3_object(client, path: str, local_file_path: str):
    # upload
    MB = 1024**2
    config = TransferConfig(
        multipart_threshold=128 * MB,
        multipart_chunksize=16 * MB,  # 156.25GiB maximum
    )
    bucket, key = split_s3_path(path)
    client.upload_file(local_file_path, bucket, key, Config=config)


def list_s3_objects(client, path: str, recursive=False, is_prefix=False, limit=0):
    for content in list_s3_objects_detailed(client, path, recursive, is_prefix, limit):
        yield content[0]


def paginated_s3_objects_with_details(client, path: str, recursive=False, is_prefix=False, limit=0, page_size=100, page_no=1):
    if not path.endswith("/") and not is_prefix:
        path += "/"
    bucket, prefix = split_s3_path(path)  # 确保 split_s3_path 返回的 prefix 是字符串类型

    marker = None
    item_yielded = 0
    current_page = 1

    while True:
        operation_parameters = {
            "Bucket": bucket,
            "Prefix": prefix,  # 这里 Prefix 应直接是一个字符串
            "MaxKeys": page_size,
            "Delimiter": "/" if not recursive else None,
        }
        if marker:
            operation_parameters["Marker"] = marker

        result = client.list_objects(**operation_parameters)
        contents = result.get("Contents", [])
        common_prefixes = result.get("CommonPrefixes", [])
        next_marker = result.get("NextMarker")

        if current_page == page_no:
            for content in contents:
                if not content["Key"].endswith("/"):
                    yield (f"s3://{bucket}/{content['Key']}", content, "file")
                    item_yielded += 1
                    if item_yielded == page_size:
                        return

            for _prefix in common_prefixes:
                yield (f"s3://{bucket}/{_prefix['Prefix']}", _prefix, "directory")
                item_yielded += 1
                if item_yielded == page_size:
                    return

        if not next_marker or item_yielded == page_size:
            break

        marker = next_marker
        current_page += 1


def list_s3_objects_detailed(client, path: str, recursive=False, is_prefix=False, limit=0):
    if limit > 1000:
        raise Exception("limit greater than 1000 is not supported.")
    if not path.endswith("/") and not is_prefix:
        path += "/"
    bucket, prefix = split_s3_path(path)
    marker = None
    while True:
        list_kwargs = dict(MaxKeys=1000, Bucket=bucket, Prefix=prefix)
        if limit > 0:
            list_kwargs["MaxKeys"] = limit
        if not recursive:
            list_kwargs["Delimiter"] = "/"
        if marker:
            list_kwargs["Marker"] = marker
        response = client.list_objects(**list_kwargs)
        marker = None
        if not recursive:
            common_prefixes = response.get("CommonPrefixes", [])
            for cp in common_prefixes:
                yield (f"s3://{bucket}/{cp['Prefix']}", cp)
            if common_prefixes:
                marker = common_prefixes[-1]["Prefix"]
        contents = response.get("Contents", [])
        for content in contents:
            if not content["Key"].endswith("/"):
                yield (f"s3://{bucket}/{content['Key']}", content)
        if contents:
            last_key = contents[-1]["Key"]
            if not marker or last_key > marker:
                marker = last_key
        if limit or not response.get("IsTruncated") or not marker:
            break


def delete_s3_object(client, path: str, dry_run=True, verbose=False):
    bucket, key = split_s3_path(path)
    if verbose:
        print(f"deleting object: s3://{bucket}/{key}")
    if not dry_run:
        client.delete_object(Bucket=bucket, Key=key)


def delete_s3_objects(client, path: str, dry_run=True, verbose=False):
    from concurrent.futures import ThreadPoolExecutor

    bucket, _ = split_s3_path(path)
    objects = list_s3_objects_detailed(client, path, recursive=True)

    def delete_key(idx, key):
        if verbose:
            print(f"deleting object [{idx}]: s3://{bucket}/{key}")
        if not dry_run:
            client.delete_object(Bucket=bucket, Key=key)

    with ThreadPoolExecutor(max_workers=20) as pool:
        for idx, (_, obj) in enumerate(objects):
            pool.submit(delete_key, idx, obj["Key"])


class ClientCache:
    def __init__(self):
        self.cache = {}

    def get_client(self, path):
        bucket, _ = split_s3_path(path)
        if not bucket:
            return get_s3_client(path)
        if bucket not in self.cache:
            self.cache[bucket] = get_s3_client(path)
        return self.cache[bucket]
