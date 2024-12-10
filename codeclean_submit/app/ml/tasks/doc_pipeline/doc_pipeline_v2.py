import hashlib
import os
import signal
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import click
import ray
import ray.actor
import ray.exceptions
from loguru import logger
from magic_pdf.libs import version as magic_pdf_version
from magic_pdf.pipe.UNIPipe import UNIPipe

from app.common.json_util import json_loads
from app.common.kafka_ext import get_kafka_python_consumer
from app.common.s3 import S3DocWriter, is_s3_object_exists, read_s3_lines, read_s3_object_bytes


class TimeOutError(Exception):
    pass


@dataclass
class Config:
    path: str
    group_id: str
    max_actor_num: int
    timeout: int


@dataclass
class Msg:
    input_path: str
    output_path: str


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def get_ext(path: str):
    filename = os.path.basename(path)
    parts = filename.split(".")
    if len(parts) > 1 and parts[0]:
        return parts[-1]
    return "txt"


@ray.remote(num_gpus=1)
class ProcessActor:
    def __init__(self, timeout: int = 0):
        self.timeout = timeout

        os.environ["YOLO_VERBOSE"] = "False"
        signal.signal(signal.SIGALRM, timeout_handler)

    def __set_extra_info(self, pdf_info, key, val):
        if "remark" not in pdf_info:
            pdf_info["remark"] = {}
        pdf_info["remark"][key] = val
        return pdf_info

    def process(self, msg: Msg):
        input_path = msg.input_path
        output_path = msg.output_path
        start_time = datetime.now()
        logger.info(f"Start process {input_path=} {output_path=}")
        logger.info(f"GPU: {ray.get_gpu_ids()}")
        __tmp_dir = os.path.join(".", "s3_upload")

        sha256 = hashlib.sha256()
        sha256.update(output_path.encode())
        filename = f"{sha256.hexdigest()}.{get_ext(output_path)}"
        filepath = os.path.join(__tmp_dir, filename)
        lines = 0
        if os.path.exists(filepath):
            with open(filepath) as f:
                lines = sum(1 for _ in f)

        writer = S3DocWriter(output_path, filename=filename, flush=True)
        for i, line in enumerate(read_s3_lines(input_path)):
            if i < lines:
                continue
            try:
                signal.alarm(self.timeout)
                info = json_loads(line)
                pdf_path = info["path"]
                pdf_bytes = read_s3_object_bytes(pdf_path)

                pdf = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer=None)
                pdf.pipe_classify()
                pdf.pipe_analyze()

                info["doc_layout_result"] = pdf.model_list

            except Exception as e:
                self.__set_extra_info(info, "__error", str(e))
                logger.error(f"process {input_path=} {output_path=}, line: {i}, gpu id: {ray.get_gpu_ids()}, error: {e}")

            finally:
                self.__set_extra_info(info, "__inference_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                self.__set_extra_info(info, "__mineru_inference_version", magic_pdf_version.__version__)
                info["__unimernet_version"] = "0.1.6"
                signal.alarm(0)

                writer.write(info)

            if i % 100 == 0:
                logger.info(f"process {input_path=} {output_path=}, line: {i}, time: {datetime.now()-start_time}")

        writer.flush()
        logger.info(f"Finish process {input_path=} {output_path=}, time: {datetime.now()-start_time}")
        return i + 1


class ProcessActorPool:
    def __init__(self, config: Config):
        self.max_actor_num = config.max_actor_num
        self.timeout = config.timeout

        self._idle_actors = deque()
        self._running_jobs = []
        self._job_to_actor = {}
        self._job_to_msg = {}

        self._msgs = deque()

        self.consumer = get_kafka_python_consumer(config.path, config.group_id)

    def run(self):
        _job_count = 0
        _job_success_count = 0
        _job_fail_count = 0
        _last_check_time = datetime.now()
        _total_success_pdf_count = 0

        while True:
            if not self._msgs:
                msgs = self.consumer.consume(timeout=10)
                if not msgs:
                    logger.warning("no msg")
                for msg in msgs:
                    if msg.error():
                        logger.error(msg.error())
                        continue
                    self._msgs.append(msg)

            if self._msgs and len(self._idle_actors) == 0:
                self.scale_up()
            elif not self._msgs and self._idle_actors:
                self.scale_down()

            completed_jobs, self._running_jobs = ray.wait(self._running_jobs, timeout=1)

            for job in completed_jobs:
                msg = self._job_to_msg.pop(job)
                actor = self._job_to_actor.pop(job)
                try:
                    pdf_count = ray.get(job)
                    self._idle_actors.append(actor)
                    self.consumer.commit(msg)
                    _job_success_count += 1
                    _total_success_pdf_count += pdf_count
                except ray.exceptions.ActorDiedError:
                    logger.warning(f"[Actor died]: retry {msg.value()}")
                    ray.kill(actor)
                    self._msgs.appendleft(msg)
                    _job_count -= 1
                except ray.exceptions.RayTaskError as e:
                    _job_fail_count += 1
                    logger.error(f"msg failed: {msg.value()}")
                    logger.exception(e)
                except Exception as e:
                    _job_fail_count += 1
                    logger.error(f"system error: {msg.value()}")
                    logger.exception(e)

            if self._msgs and self._idle_actors:
                msg = self._msgs.popleft()
                actor = self._idle_actors.popleft()
                msg_dict = json_loads(msg.value())
                msg_obj = Msg(**msg_dict)
                if not is_s3_object_exists(msg_obj.input_path):
                    logger.warning(f"input path {msg_obj.input_path} not exists")
                    self.consumer.commit(msg)
                    continue

                if is_s3_object_exists(msg_obj.output_path):
                    logger.warning(f"output path {msg_obj.output_path} already exists")
                    self.consumer.commit(msg)
                    continue

                job = actor.process.remote(msg_obj)
                self._running_jobs.append(job)
                self._job_to_actor[job] = actor
                self._job_to_msg[job] = msg
                _job_count += 1

            _interval = datetime.now() - _last_check_time
            if _interval > timedelta(minutes=10):
                logger.info(
                    f"total job count: {_job_count}, success: {_job_success_count}, fail: {_job_fail_count}, total success pdf count: {_total_success_pdf_count}"
                )
                _last_check_time = datetime.now()

            time.sleep(1)

    def scale_up(self):
        if self.max_actor_num <= 0 or len(self._running_jobs) + len(self._idle_actors) < self.max_actor_num:
            gpu_num = int(ray.available_resources().get("GPU") or 0)
            if gpu_num > 0:
                actor = ProcessActor.remote(timeout=self.timeout)
                self._idle_actors.append(actor)
                logger.info(f"scale up 1 actors, now {len(self._idle_actors)+len(self._running_jobs)} actors")

    def scale_down(self):
        while self._idle_actors:
            actor = self._idle_actors.popleft()
            ray.kill(actor)

        logger.info(f"scale down {len(self._idle_actors)} actors, now {len(self._running_jobs)} actors")


@click.command()
@click.option("--path", type=str, default="kafka://kafka_ali/doc_pipeline_test")
@click.option("--group_id", type=str, default="doc_pipeline_test")
@click.option("--max_actor_num", type=int, default=0)
@click.option("--timeout", type=int, default=0)
def command(path, group_id, max_actor_num, timeout):
    logger.info(f"Start doc pipeline v2 with {path=}, {group_id=}, {max_actor_num=}, {timeout=}")
    ray.init()
    config = Config(path=path, group_id=group_id, max_actor_num=max_actor_num, timeout=timeout)
    p = ProcessActorPool(config)
    p.run()


if __name__ == "__main__":
    command()
