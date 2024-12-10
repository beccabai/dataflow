from app.common.const import *
from app.common.json_util import json_dumps
from app.common.s3.path import ensure_s3_path
from app.common.s3.retry import put_s3_object_with_retry


def __write_mark_in_path(mark: str, path: str, body_text: str = ""):
    prefix = ensure_s3_path(path)
    mark_file = f"{prefix.rstrip('/')}/{mark}"
    body = body_text.encode("utf-8")
    put_s3_object_with_retry(mark_file, body)


def mark_failure_in_s3(output_path: str, task_info: dict):
    body_text = json_dumps(task_info)
    __write_mark_in_path(FAILURE_MARK_FILE, output_path, body_text)


def mark_success_in_s3(output_path: str, task_info: dict):
    body_text = json_dumps(task_info)
    __write_mark_in_path(SUCCESS_MARK_FILE, output_path, body_text)


def mark_reserve_in_s3(output_path: str):
    __write_mark_in_path(RESERVE_MARK_FILE, output_path)


def mark_deleted_in_s3(output_path: str):
    __write_mark_in_path(DELETED_MARK_FILE, output_path)


def mark_summary_in_s3(output_path: str, summary_info: dict):
    summary_pairs = []
    for k, v in summary_info.items():
        if k == "sub_paths" or is_flag_field(k) or is_acc_field(k) or is_qa_error(k) or is_qa_validation_summary(k):
            continue
        if type(v) == dict:
            v = v["sum"]
        summary_pairs.append(f"_{k}_{v}")

    mark_file = f'{SUMMARY_MARK_FILE}{"".join(summary_pairs)}'
    body_text = json_dumps(summary_info)
    __write_mark_in_path(mark_file, output_path, body_text)
