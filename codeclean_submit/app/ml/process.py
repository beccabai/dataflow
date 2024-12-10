import time
from typing import Callable, Union

from app.common.s3 import S3DocWriter, is_s3_path
from app.ml.loader import LoaderLoader
from app.ml.utils.attribute import LocalAttributeWriter
from app.ml.utils.local_fs import LocalDocWriter


def process_one_file(inference_fn: Callable, loader, output_file: str):
    writer = S3DocWriter(output_file, skip_loc=True)
    f_begin = begin = time.time()
    f_cnt = 0

    for doc_infos, batch in loader:
        batch_sz = len(doc_infos)

        now = time.time()
        l_cost = round(now - begin, 6)
        begin = now

        results = inference_fn(batch)

        now = time.time()
        m_cost = round(now - begin, 6)
        begin = now

        for idx, doc_info in enumerate(doc_infos):
            if doc_info.get("error"):
                continue
            d = {**doc_info, **results[idx]}
            writer.write(d)

        now = time.time()
        w_cost = round(now - begin, 6)
        f_cost = round(now - f_begin, 6)
        begin = now

        hl = f"{f_cnt}+{batch_sz}"
        f_cnt += batch_sz
        b_qps = round(batch_sz / max(m_cost, 1e-6), 6)
        f_qps = round(f_cnt / max(f_cost, 1e-6), 6)

        print(f"{hl} | {l_cost} | {m_cost} | {w_cost} | {b_qps} | {f_qps}")

    writer.flush()


def process_in_loop(
    inference_fn: Callable,
    ll: LoaderLoader,
    callback: Union[Callable, None] = None,
    writer_config: dict = {},
    verbose=True,
):
    print("LOOP ENTER")

    fl_cost = 0
    a_begin = begin = time.time()
    a_cnt = 0
    last_report_time = a_begin

    while True:
        loader_and_info = ll.next()
        if not loader_and_info:
            print("loader is None")
            break

        loader, info = loader_and_info
        output_file = info["output_file"]
        if is_s3_path(output_file):
            writer = S3DocWriter(path=output_file, skip_loc=True)
        elif output_file.endswith("mmap.npy"):
            writer = LocalAttributeWriter(path=output_file, **writer_config)
        else:
            writer = LocalDocWriter(path=output_file)

        now = time.time()
        ll_cost = round(now - begin, 2)
        f_begin = begin = now
        f_cnt = 0

        for doc_infos, batch in loader:
            batch_sz = len(doc_infos)

            now = time.time()
            l_cost = round(now - begin, 6)
            begin = now

            results = inference_fn(batch)

            now = time.time()
            m_cost = round(now - begin, 6)
            begin = now

            for idx, doc_info in enumerate(doc_infos):
                if doc_info.get("error"):
                    continue
                d = {**doc_info, **results[idx]}
                writer.write(d)

            now = time.time()
            w_cost = round(now - begin, 6)
            f_cost = round(now - f_begin, 6)
            a_cost = round(now - a_begin, 6)
            begin = now

            hl = f"{a_cnt}+{f_cnt}+{batch_sz}"
            f_cnt += batch_sz
            b_qps = round(batch_sz / max(m_cost, 1e-6), 6)
            f_qps = round(f_cnt / max(f_cost, 1e-6), 6)
            a_qps = round((a_cnt + f_cnt) / max(a_cost, 1e-6), 6)

            if verbose or (now - last_report_time > 10):
                last_report_time = now
                print(f"{hl} | {fl_cost} | {ll_cost} | {l_cost} | {m_cost} | {w_cost} | {b_qps} | {f_qps} | {a_qps}")

            ll_cost = 0
            fl_cost = 0
        writer.flush()

        if callback is not None:
            try:
                callback(info)
            except Exception as e:
                print(f"Callback failed: {e}")

        now = time.time()
        fl_cost = round(now - begin, 2)
        begin = now

        a_cnt += f_cnt

    print("LOOP EXIT")
