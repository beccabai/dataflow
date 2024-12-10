from app.common.const import *
from app.common.s3.client import get_s3_client, head_s3_object, list_s3_objects
from app.common.s3.path import ensure_s3_path, ensure_s3a_path


def is_s3_empty_path(output_path: str) -> bool:
    check_path = output_path.rstrip("/") + "/"
    client = get_s3_client(check_path)
    contents = list_s3_objects(client, check_path, recursive=True, limit=10)
    for c in contents:
        if c.endswith(RESERVE_MARK_FILE):
            continue
        return False
    return True


def is_s3_success_path(input_path: str) -> bool:
    client = get_s3_client(input_path)

    def is_success_path(path: str) -> bool:
        check_path = path.rstrip("/") + "/"
        for mark in [SUCCESS_MARK_FILE, SUCCESS_MARK_FILE2]:
            if head_s3_object(client, check_path + mark):
                return True
        return False

    if is_success_path(input_path):
        return True

    sub_dirs = list(list_s3_objects(client, input_path))

    # fmt: off
    return len(sub_dirs) > 0 and \
        all([dir.endswith("/") for dir in sub_dirs]) and \
        all([is_success_path(dir) for dir in sub_dirs])
    # fmt: on


def is_s3_path_exists(path: str) -> bool:
    check_path = path.rstrip("/") + "/"
    client = get_s3_client(check_path)
    contents = list_s3_objects(client, check_path, limit=10)
    for c in contents:
        return True
    return False


def is_s3_object_exists(path: str) -> bool:
    client = get_s3_client(path)
    return bool(head_s3_object(client, path))


def detect_s3_multi_layer_path(input_path: str) -> str:
    if "*" in input_path:
        return ensure_s3a_path(input_path)

    check_path = ensure_s3_path(input_path.rstrip("/") + "/")
    client = get_s3_client(check_path)
    contents = list_s3_objects(client, check_path, recursive=True, limit=1000)

    relative_path = None
    for c in contents:
        sub_path = c[len(check_path) :]
        if sub_path.startswith("_") or sub_path.startswith("."):
            continue
        relative_path = sub_path
        break

    if not relative_path:
        raise Exception(f"cannot find file in input path [{input_path}].")

    num_relative_parts = len(relative_path.split("/"))
    if num_relative_parts > 1:
        check_path += "*/" * (num_relative_parts - 1)

    return ensure_s3a_path(check_path)
