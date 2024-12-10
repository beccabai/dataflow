from datetime import timezone

from app.common.json_util import json_dumps, json_print
from app.common.s3.client import (
    get_s3_client,
    list_s3_objects,
    list_s3_objects_detailed,
)
from app.common.s3.other import get_s3_presigned_url
from app.common.s3.read import read_s3_row, read_s3_rows
from app.common.s3.retry import head_s3_object_with_retry


def _format_datetime(dt):
    if not dt:
        return ""
    dt = dt.replace(tzinfo=timezone.utc).astimezone(tz=None)  # localtime
    return dt.strftime("%y-%m-%d %H:%M:%S %Z")


def _format_size(size):
    if size is None:
        return ""
    size = str(size)
    parts = []
    while len(size):
        part_size = 3
        if not parts and len(size) % part_size:
            part_size = len(size) % part_size
        parts.append(size[:part_size])
        size = size[part_size:]
    return ",".join(parts)


def _format_detail(detail):
    path, obj = detail
    if path.endswith("/"):
        return f"{'DIR'.rjust(53)}  {path}"
    tm = _format_datetime(obj.get("LastModified"))
    sz = _format_size(obj.get("Size") or obj.get("ContentLength", 0))
    owner = obj.get("Owner", {}).get("ID", "")
    return f"{tm} {sz.rjust(15)} {owner.rjust(15)}  {path}"


def head(path):
    obj_head = head_s3_object_with_retry(path)
    if obj_head is not None:
        print(json_dumps(obj_head, indent=2, default=str))


def cat(path, limit=1, show_loc=False):
    if "?bytes=" in path:
        row = read_s3_row(path)
        if row is not None:
            if show_loc:
                print(row.loc)
            json_print(row)
        return
    for row in read_s3_rows(path, use_stream=True, limit=limit):
        if show_loc:
            print(row.loc)
        json_print(row)


def ls(path, limit=100):
    client = get_s3_client(path)
    for obj in list_s3_objects(client, path, limit=limit):
        print(obj)


def ls_r(path, limit=100):
    client = get_s3_client(path)
    for item in list_s3_objects(client, path, True, True, limit):
        print(item)


def ll(path, limit=100):
    client = get_s3_client(path)
    for detail in list_s3_objects_detailed(client, path, limit=limit):
        print(_format_detail(detail))


def ll_r(path, limit=100):
    client = get_s3_client(path)
    for detail in list_s3_objects_detailed(client, path, True, True, limit):
        print(_format_detail(detail))


def download(path):
    print(get_s3_presigned_url(path))
