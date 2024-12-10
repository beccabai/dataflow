import os

import ray
from ray.util.queue import Queue

from app.ml.kafka_ray import KafkaRayQueuePipe

PATH = "kafka://kafka_3/pp_imglabel_files"
GROUP_ID = "pipeline_prod"

ray.init(address="ray://10.140.52.159:10001")
gpu_num = int(ray.cluster_resources().get("GPU") or 2)


@ray.remote(num_gpus=1, max_retries=5, retry_exceptions=True)
def process(task_queue: Queue, finish_queue: Queue):
    os.chdir("/tmp")

    from app.common.kafka_ext import KafkaWriter
    from app.common.s3 import head_s3_object_with_retry
    from app.ml.dataset import S3ImageH5Dataset
    from app.ml.img_cls import ImgClassifier
    from app.ml.loader import InfoQueueLoaderLoader
    from app.ml.process import process_in_loop
    from app.pipeline.const import DATA_FILE_PATH, LABEL_FILE_PATH

    def write_message(info: dict):
        writer = KafkaWriter("kafka://kafka_3/pp_mmimport_files")
        writer.write({"file": info["output_file"]})
        writer.flush()

    def process_callback(info: dict):
        write_message(info)
        finish_queue.put(info)

    class MyLoaderLoader(InfoQueueLoaderLoader):
        def should_skip(self, info: dict):
            file = str(info.get("file") or "")
            if not file:
                print(f"field [file] is missing, skip")
                finish_queue.put(info)
                return True
            assert file.startswith(DATA_FILE_PATH) and file.endswith(".h5")
            label_file = LABEL_FILE_PATH + file[len(DATA_FILE_PATH) : -len(".h5")] + ".jsonl"
            info["output_file"] = label_file
            if head_s3_object_with_retry(label_file):
                write_message(info)
                finish_queue.put(info)
                return True
            if not head_s3_object_with_retry(file):
                finish_queue.put(info)
                return True
            return False

    model = ImgClassifier()
    loader_loader = MyLoaderLoader(
        task_queue,
        S3ImageH5Dataset,
        model.preprocess,
        model.collate_fn,
    )
    loader_loader.start()
    process_in_loop(model.inference, loader_loader, process_callback)


class ProcessMonitor:
    def __init__(self):
        self.ids = []

    def set_pipe(self, pipe: KafkaRayQueuePipe):
        self.pipe = pipe

    def monitor(self):
        done_ids, self.ids = ray.wait(self.ids, num_returns=gpu_num, timeout=0)
        for done_id in done_ids:
            try:
                print(ray.get(done_id))
            except Exception as e:
                print(e)

        while len(self.ids) < gpu_num:
            self.ids.append(process.remote(self.pipe.task_queue, self.pipe.finish_queue))


def main():
    mon = ProcessMonitor()
    pipe = KafkaRayQueuePipe(PATH, GROUP_ID, monitor_fn=mon.monitor)
    mon.set_pipe(pipe)
    pipe.start()


if __name__ == "__main__":
    main()
