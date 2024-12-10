import os
from datetime import datetime

import ray
from loguru import logger
from magic_pdf.libs import version as magic_pdf_version
from magic_pdf.pipe.UNIPipe import UNIPipe
from ray.util.queue import Queue

from app.common.json_util import json_loads
from app.common.s3 import S3DocWriter, read_s3_lines, read_s3_object_bytes
from app.ml.kafka_ray import KafkaRayQueuePipe

ray.init(address=os.environ["ip_head"])
gpu_num = int(ray.cluster_resources().get("GPU") or 2)


def __set_extra_info(pdf_info, key, val):
    if "remark" not in pdf_info:
        pdf_info["remark"] = {}
    pdf_info["remark"][key] = val
    return pdf_info


@ray.remote(num_gpus=1, max_retries=5, retry_exceptions=True)
def process(task_queue: Queue, finish_queue: Queue, pending: bool = True):
    os.environ["YOLO_VERBOSE"] = "False"

    while True:
        logger.info("Waiting for task...")
        msg = task_queue.get(block=pending)
        if msg is None:
            break

        input_path: str = msg["input_path"]
        logger.info(f"Start processing {input_path}...")
        if "output_path" in msg:
            output_path = msg["output_path"]
        elif "output_dir" in msg:
            output_path = msg["output_dir"].strip("/") + input_path.split("/")[-1]
        else:
            output_path = "s3://llm-pdf-text/books/processing/test/" + input_path.split("/")[-1]

        logger.info(f"Output path: {output_path}")

        writer = S3DocWriter(output_path)

        for i, line in enumerate(read_s3_lines(input_path)):
            info = json_loads(line)
            try:
                # 读取pdf路径
                pdf_path = info["path"]
                logger.info(f"line {i}: {pdf_path}")

                # 读取pdf bytes
                pdf_bytes = read_s3_object_bytes(pdf_path)

                # 对pdf进行处理
                pdf = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer=None)
                pdf.pipe_classify()
                pdf.pipe_analyze()

                # 将处理结果写回doc
                info["doc_layout_result"] = pdf.model_list

            except Exception as e:
                __set_extra_info(info, "__error", str(e))
                logger.exception(e)

            finally:
                logger.info(f"finish processing {pdf_path}")
                __set_extra_info(info, "__inference_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                __set_extra_info(info, "__mineru_inference_version", magic_pdf_version.__version__)
                # TODO:增加其余meta信息

                writer.write(info)

        writer.flush()
        # 写回finish_queue
        finish_queue.put(msg)


class ProcessMonitor:
    def __init__(self):
        self.ids = []

    def set_pipe(self, pipe: KafkaRayQueuePipe):
        self.pipe = pipe

    def monitor(self):
        global gpu_num
        done_ids, self.ids = ray.wait(self.ids, num_returns=gpu_num, timeout=0)
        for done_id in done_ids:
            try:
                print(ray.get(done_id))
            except Exception as e:
                print(e)

        gpu_num = int(ray.cluster_resources().get("GPU") or 2)
        while len(self.ids) < gpu_num:
            self.ids.append(process.remote(self.pipe.task_queue, self.pipe.finish_queue, pending=True))


PATH = "kafka://kafka_ali/doc_pipeline"
GROUP_ID = "doc_pipeline"


def main():
    logger.info("Start doc pipeline...")
    logger.info(f"Kafka path: {PATH}, group_id: {GROUP_ID}")
    mon = ProcessMonitor()
    pipe = KafkaRayQueuePipe(PATH, GROUP_ID, monitor_fn=mon.monitor)
    mon.set_pipe(pipe)
    pipe.start()


if __name__ == "__main__":
    main()
