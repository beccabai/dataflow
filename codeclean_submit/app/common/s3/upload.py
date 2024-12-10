import bz2
import gzip
import io
import os
import time
import uuid
from typing import Dict, Iterator, Union

from app.common.const import *
from app.common.json_util import json_encode, json_loads
from app.common.s3.path import ensure_s3_path, is_s3_path
from app.common.s3.retry import put_s3_object_with_retry, s3_upload_with_retry
from app.common.s3.write import _compressions
from app.qa.common.const import QA_ERROR_RULE_PASS_MAP


class S3UploadAcc:
    def __init__(self, sc):
        from app.common.spark_ext import DictAccumulatorParam

        self.acc = sc.accumulator({}, DictAccumulatorParam())

    def incr(self, field: str, sub_path: str, value: Union[int, list]):
        self.acc.add({f"_:{field}": value})
        if sub_path:
            self.acc.add({f"{sub_path}:{field}": value})

    def to_dict(self) -> dict:
        acc_value: Dict[str, int] = self.acc.value  # type: ignore
        sub_paths = {}

        for key, value in acc_value.items():
            sub_path, field = key.split(":")
            sub_path_dict = sub_paths.get(sub_path, {})
            sub_path_dict[field] = value
            sub_paths[sub_path] = sub_path_dict

        d = {**sub_paths.get("_", {})}
        # 这里增加qa质检的统计, 不包含sub_paths内的统计，格式如下：
        # "qa_validation_result": {
        #       "total_errors": 100,
        #       "ERROR_RULE_NO_PUNC":{"count": 20, "ratio": 0.001, "pass": true},
        #       "ERROR_RULE_DOC_REPEAT":{"count": 80, "ratio": 0.004, "pass": false}
        #   }
        if is_summary_contain_qa_error(d):
            total_docs = d.get("rows", 0) + d.get("dropped", 0)
            qa_validation_result = {}
            qa_validation_result["total_docs"] = total_docs
            qa_validation_result["total_errors"] = d.get("error_type")
            for key, value in d.items():
                if is_qa_error(key):
                    qa_validation_result[key] = {}
                    ratio = value / total_docs
                    qa_validation_result[key]["count"] = value
                    qa_validation_result[key]["ratio"] = ratio
                    qa_validation_result[key]["pass"] = ratio <= QA_ERROR_RULE_PASS_MAP.get(key, 0)
            d["qa_validation_result"] = qa_validation_result

        d["sub_paths"] = dict(sorted(filter(lambda i: i[0] != "_", sub_paths.items())))
        return d


def upload_to_s3(
    output_path: str,
    ext: str,
    acc: S3UploadAcc,
    track_ratio: float,
    skip_loc=False,
    partition_key=FIELD_SUB_PATH,
    prefix="part",
    write_tokens_meta=False,
    write_ids_file=False,
    skip_main_file=False,
    write_attr_list={},
    process_func=None,
    compression="",
    plaintext=False,
    is_cal_token=False,
):
    from pyspark import TaskContext
    from pyspark.sql import Row

    compression = _compressions.get(compression)
    # uid = str(uuid.uuid4())[:8]
    uid = int(time.time() * 65536).to_bytes(6, byteorder="big").hex()

    def mix(d: dict, row: Row, changed=False) -> dict:
        ret = {**d}
        for key, val in row.asDict().items():
            if key != "value" and key not in d:
                ret[key] = val
        if changed:
            ret["changed"] = True
        return ret

    def get_attr_val(d: dict, attr_path: str):
        if d.get(attr_path):
            return d[attr_path]
        parts = attr_path.split(".", 1)
        if len(parts) > 1 and isinstance(d.get(parts[0]), dict):
            return get_attr_val(d[parts[0]], parts[1])
        return 0.0  # as default value

    def handle(iter: Iterator[Row]):
        ctx = TaskContext.get()

        if ctx is None:
            raise Exception("cannot get task context.")

        tmp_upload_dir = os.path.join(".", "s3_upload")
        os.makedirs(tmp_upload_dir, exist_ok=True)

        # avoid importing pymongo if track_ratio is le zero.
        doc_tracker = None
        if track_ratio and track_ratio > 0.0:
            from app.common.track import DocTracker

            doc_tracker = DocTracker(output_path, track_ratio)

        def get_output_file(sub_path):
            if is_s3_path(sub_path):
                path = ensure_s3_path(sub_path)
                if compression and not path.endswith(f".{compression}"):
                    path = f"{path}.{compression}"
                return path

            file_prefix = prefix or "part"
            if file_prefix == "{split}":
                if "/val/" in f"/{sub_path}/":
                    file_prefix = "val"
                elif "/test/" in f"/{sub_path}/":
                    file_prefix = "test"
                else:
                    file_prefix = "train"

            output_name = f"{file_prefix}-{uid}-{str(ctx.partitionId()).zfill(6)}.{ext}"
            if compression:
                output_name = f"{output_name}.{compression}"
            if sub_path:
                output_file = f"{output_path.rstrip('/')}/{sub_path}/{output_name}"
            else:
                output_file = f"{output_path.rstrip('/')}/{output_name}"
            return output_file

        if process_func and callable(process_func):
            iter = process_func(iter)

        # sub_path -> (tmp_fh, tmp_filename, offset, output_file, tokens_meta)
        tmp_files = {}
        try:
            for row in iter:
                if plaintext:
                    if partition_key and partition_key in row:
                        sub_path = str(row[partition_key] or "").strip("/")
                    else:
                        sub_path = ""

                    if "output_file" in row and row["output_file"]:
                        tmp_file_key = str(row["output_file"] or "").strip("/")
                    else:
                        tmp_file_key = sub_path

                    tmp_file = tmp_files.get(tmp_file_key)
                    if not tmp_file:
                        tmp_filename = os.path.join(tmp_upload_dir, f"{str(uuid.uuid4())}.{ext}")
                        tmp_fh = open(tmp_filename, "ab")
                        tmp_file = (tmp_fh, tmp_filename, [0], get_output_file(tmp_file_key), {})
                        tmp_file[-1]["sub_path"] = sub_path
                        tmp_files[tmp_file_key] = tmp_file

                    # TODO: line shouldn't contains newline.
                    line = str(row.value) + "\n"
                    line_bytes = line.encode("utf-8")

                    tmp_fh = tmp_file[0]
                    tmp_fh.write(line_bytes)

                    acc.incr("rows", "", 1)
                    acc.incr("bytes", "", [len(line_bytes)])
                    continue

                try:
                    d = json_loads(str(row.value))
                except Exception as e:
                    print(row)
                    raise e

                if partition_key:
                    sub_path = (mix(d, row).get(partition_key) or "").strip("/")
                else:
                    sub_path = ""

                # handle flags and metrics
                for key, val in d.items():
                    if is_flag_field(key) and val == True:
                        acc.incr(key, sub_path, 1)
                    elif is_acc_field(key) and type(val) == int and val >= 0:
                        acc.incr(key, sub_path, [val])

                if d.get("dropped"):
                    if doc_tracker:
                        doc_tracker.track(mix(d, row), sub_path)
                    acc.incr("dropped", sub_path, 1)
                    continue

                changed = False
                if "changed" in d:
                    changed = bool(d["changed"])
                    del d["changed"]

                if changed:
                    acc.incr("changed", sub_path, 1)

                tmp_file_key = mix(d, row).get("output_file") or sub_path
                tmp_file = tmp_files.get(tmp_file_key)
                if not tmp_file:
                    tmp_filename = os.path.join(tmp_upload_dir, f"{str(uuid.uuid4())}.{ext}")
                    tmp_fh = open(tmp_filename, "ab")
                    tmp_file = (tmp_fh, tmp_filename, [0], get_output_file(tmp_file_key), {})
                    tmp_file[-1]["sub_path"] = sub_path
                    tmp_files[tmp_file_key] = tmp_file

                tmp_fh, _, offset, output_file, extra_info = tmp_file

                if not skip_loc and "doc_loc" in d:
                    track_loc = d.get("track_loc") or []
                    track_loc.append(d["doc_loc"])
                    d["track_loc"] = track_loc

                if compression == "gz":
                    if not skip_loc and FIELD_ID in d:
                        d["doc_loc"] = f"{output_path}?bytes={offset[0]},0"
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
                        f.write(json_encode(d))
                    doc_bytes = buf.getvalue()

                elif compression == "bz2":
                    if not skip_loc and FIELD_ID in d:
                        d["doc_loc"] = f"{output_path}?bytes={offset[0]},0"
                    buf = io.BytesIO()
                    with bz2.BZ2File(buf, mode="wb") as f:
                        f.write(json_encode(d))
                    doc_bytes = buf.getvalue()

                else:
                    doc_bytes = json_encode(d)

                    # add doc_loc if doc has id
                    if not skip_loc and FIELD_ID in d:
                        doc_len, last_len = len(doc_bytes), 0
                        while doc_len != last_len:
                            d["doc_loc"] = f"{output_file}?bytes={offset[0]},{doc_len}"
                            doc_bytes = json_encode(d)
                            doc_len, last_len = len(doc_bytes), doc_len

                if not skip_main_file:
                    tmp_fh.write(doc_bytes)

                token_num = len(d.get("tokens", []))
                if write_tokens_meta:
                    tokens_meta = extra_info.setdefault("tokens_meta", [])
                    tokens_meta.append((offset[0], token_num))
                if write_ids_file:
                    id_list = extra_info.setdefault("id_list", [])
                    id_list.append(str(d.get("id", "")))
                if write_attr_list:
                    for key, val in write_attr_list.items():
                        attr_list = extra_info.setdefault(f"attr-{key}", [])
                        attr_list.append(get_attr_val(d, val))

                offset[0] += len(doc_bytes)
                acc.incr("rows", sub_path, 1)
                acc.incr("bytes", sub_path, [len(doc_bytes)])

                # handle qa error statics
                for key, val in d.items():
                    if is_qa_error(key) and isinstance(val, dict):
                        if len(val) > 0:
                            acc.incr(key, sub_path, 1)
                            for qa_key, qa_val in val.items():
                                if qa_val:
                                    acc.incr(qa_key, sub_path, 1)

                if "content" in d and type(d["content"]) is str:
                    acc.incr("cbytes", sub_path, [len(d["content"].encode("utf-8"))])
                if token_num > 0:
                    acc.incr("tokens", sub_path, [token_num])
                elif is_cal_token:
                    from app.snippets.tokenize_v7 import content_to_tokens

                    acc.incr("tokens", sub_path, len(content_to_tokens(d["content"])) + 2)

                if doc_tracker:
                    doc_tracker.track(mix(d, row, changed), sub_path)

            if doc_tracker:
                doc_tracker.flush()

            for tmp_file_key in tmp_files.keys():
                tmp_fh, tmp_filename, _, output_file, extra_info = tmp_files[tmp_file_key]
                tmp_fh.close()

                sub_path = extra_info.get("sub_path", "")

                if not skip_main_file:
                    s3_upload_with_retry(output_file, tmp_filename)
                    acc.incr("files", sub_path, 1)

                if write_tokens_meta:
                    import numpy as np

                    tokens_meta = extra_info.get("tokens_meta", [])
                    meta_np = np.array(tokens_meta, dtype=np.int64)
                    buffer = io.BytesIO()
                    np.save(buffer, meta_np)
                    put_s3_object_with_retry(f"{output_file}.meta", buffer.getvalue())
                    acc.incr("files", sub_path, 1)

                if write_ids_file:
                    id_list = extra_info.get("id_list", [])
                    body = "".join([id + "\n" for id in id_list]).encode("utf-8")
                    put_s3_object_with_retry(f"{output_file}.ids", body)
                    acc.incr("files", sub_path, 1)

                if write_attr_list:
                    import numpy as np

                    for key, val in write_attr_list.items():
                        attr_list = extra_info.get(f"attr-{key}", [])
                        buffer = io.BytesIO()
                        np.save(buffer, np.array(attr_list, dtype=np.float16))
                        put_s3_object_with_retry(f"{output_file}.attr-{key}.npz", buffer.getvalue())
                        acc.incr("files", sub_path, 1)

        finally:
            for sub_path in tmp_files.keys():
                tmp_filename = tmp_files[sub_path][1]
                os.remove(tmp_filename)

    return handle
