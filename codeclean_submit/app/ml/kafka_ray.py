import time
from typing import Callable, Union

from ray.util.queue import Queue

from app.common.json_util import json_loads
from app.common.kafka_ext import get_kafka_python_consumer


class KafkaRayQueuePipe:
    def __init__(
        self,
        path: str,
        group_id: str,
        queue_capacity=100,
        consume_timeout=2,
        session_timeout=20,
        pipe_interval=5,
        monitor_fn: Union[Callable, None] = None,
        verbose=True,
    ):
        """path: kafka://kafka_3/topic_name"""
        self.path = path
        self.group_id = group_id
        self.queue_capacity = queue_capacity
        self.consume_timeout = consume_timeout
        self.session_timeout = session_timeout
        self.pipe_interval = pipe_interval
        self.monitor_fn = monitor_fn
        self.verbose = verbose
        self.task_queue = Queue()
        self.finish_queue = Queue()

    def start(self):
        consumer = get_kafka_python_consumer(self.path, self.group_id)
        # TODO: close consumer if start function can exit.

        last_consume_time = 0
        seq_id = 0
        commit_id = 0
        messages = {}
        finished = {}
        queue_memo = []

        def enqueue(seq_id: int, info: dict):
            info["seq_id"] = seq_id
            queue_memo.append(seq_id)
            self.task_queue.put(info)

        while True:
            now = time.time()
            queue_sz = self.task_queue.qsize()
            queue_memo = queue_memo[-queue_sz:]
            queue_gap = max(0, self.queue_capacity - queue_sz)
            session_ttl = last_consume_time + self.session_timeout - now

            # write task_queue
            if queue_gap > 0 or session_ttl < 0:
                last_consume_time = now
                for msg in consumer.consume(
                    queue_gap,
                    timeout=self.consume_timeout,
                ):
                    if msg.error():
                        print(msg.error())
                        continue
                    info = json_loads(msg.value())
                    messages[seq_id] = msg
                    enqueue(seq_id, info)
                    seq_id += 1

            # read finish_queue
            while not self.finish_queue.empty():
                info = self.finish_queue.get()
                if info["seq_id"] >= commit_id:
                    finished[info["seq_id"]] = True

            # commit finished messages
            while finished.get(commit_id):
                finished.pop(commit_id)
                msg = messages.pop(commit_id)
                consumer.commit(message=msg)
                commit_id += 1

            # commit gap is too large, put task again.
            if ((seq_id - commit_id) > (self.queue_capacity * 3)) and (commit_id not in queue_memo):
                msg = messages[commit_id]
                info = json_loads(msg.value())
                enqueue(commit_id, info)
                print(f"[{self.__class__.__name__}] Re-put task id={commit_id} {info=}")

            if self.verbose:
                commit_waits = len(finished)
                print(f"[{self.__class__.__name__}] {seq_id=} {commit_id=} {queue_sz=} {commit_waits=}")

            if self.monitor_fn is not None:
                try:
                    self.monitor_fn()
                except Exception as e:
                    print(f"Monitor error: {e}")

            sleep_time = max(0, now + self.pipe_interval - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
