import random
from typing import List, Tuple, Union

from app.common.runtime import get_cluster_name
from app.common.s3.path import split_s3_path
from app.config import s3_bucket_prefixes, s3_buckets, s3_profiles

__spark_configs = {
    "spark.hadoop.fs.s3a.connection.maximum": "50",  # may no enough sometime.
    "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.list.version": "1",
    "spark.hadoop.fs.s3a.paging.maximum": "1000",
    "spark.hadoop.fs.s3a.committer.name": "directory",
}


def __get_s3_bucket_config(path: str):
    bucket = split_s3_path(path)[0] if path else ""
    bucket_config = s3_buckets.get(bucket)
    if not bucket_config:
        for prefix, c in s3_bucket_prefixes.items():
            if bucket.startswith(prefix):
                bucket_config = c
                break
    if not bucket_config:
        bucket_config = s3_profiles.get(bucket)
    if not bucket_config:
        bucket_config = s3_buckets.get("[default]")
        assert bucket_config is not None
    return bucket_config


def __get_s3_config(
    bucket_config,
    outside: bool,
    prefer_ip=False,
    prefer_auto=False,
):
    cluster = bucket_config["cluster"]
    assert isinstance(cluster, dict)

    if outside:
        endpoint_key = "outside"
    elif prefer_auto:
        endpoint_key = "auto"
    elif cluster.get("cluster") == get_cluster_name():
        endpoint_key = "inside"
    else:
        endpoint_key = "outside"

    if endpoint_key not in cluster:
        endpoint_key = "outside"

    if prefer_ip and f"{endpoint_key}_ips" in cluster:
        endpoint_key = f"{endpoint_key}_ips"

    endpoints = cluster[endpoint_key]

    if isinstance(endpoints, str):
        endpoint = endpoints
    elif isinstance(endpoints, list):
        endpoint = random.choice(endpoints)
    else:
        raise Exception(f"invalid endpoint for [{cluster}]")

    return {
        "endpoint": endpoint,
        "ak": bucket_config["ak"],
        "sk": bucket_config["sk"],
    }


def get_s3_config(path: Union[str, List[str]], outside=False):
    paths = [path] if type(path) == str else path
    bucket_config = None
    for p in paths:
        bc = __get_s3_bucket_config(p)
        if bucket_config in [bc, None]:
            bucket_config = bc
            continue
        raise Exception(f"{paths} have different s3 config, cannot read together.")
    if not bucket_config:
        raise Exception("path is empty.")
    return __get_s3_config(bucket_config, outside, prefer_ip=True)


def get_s3_spark_configs(outside=False) -> List[Tuple[str, str]]:
    ret = [(k, v) for k, v in __spark_configs.items()]
    sc_prefix = "spark.hadoop.fs.s3a"
    sc_items = {
        "ak": "access.key",
        "sk": "secret.key",
        "endpoint": "endpoint",
        "path.style.access": "path.style.access",
    }

    default_config = __get_s3_config(s3_buckets["[default]"], outside, prefer_auto=True)
    for key, sc_item in sc_items.items():
        if key not in default_config:
            continue
        ret.append((f"{sc_prefix}.{sc_item}", default_config[key]))

    for bucket, bucket_config in s3_buckets.items():
        if (bucket == "[default]") or ("_" in bucket):
            continue
        s3_config = __get_s3_config(bucket_config, outside, prefer_auto=True)
        for key, sc_item in sc_items.items():
            if key not in s3_config:
                continue
            if s3_config[key] == default_config[key]:
                continue
            ret.append((f"{sc_prefix}.bucket.{bucket}.{sc_item}", s3_config[key]))

    return ret
