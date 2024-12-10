import threading
import time
from typing import Callable, List, Type

from torch.utils.data import DataLoader

from app.common.json_util import json_loads
from app.ml.dataset import PathDataset


def identical(input):
    return input


class LoaderLoader(threading.Thread):
    sleep_seconds = 0.05

    def __init__(
        self,
        dataset_cls: Type[PathDataset],
        preprocess: Callable = identical,
        collate_fn: Callable = identical,
        batch_size=512,
        num_workers=5,
        timeout=10,
        prefetch_size=1,
        dataset_cls_kwargs={},
        **kwargs,
    ):
        threading.Thread.__init__(self, daemon=True)
        self.dataset_cls = dataset_cls
        self.preprocess = preprocess
        self.collate_fn = collate_fn
        self.batch_size = max(batch_size, 1)
        self.num_workers = max(num_workers, 0)
        self.timeout = max(timeout, 0)
        self.prefetch_size = max(prefetch_size, 1)
        self.dataset_cls_kwargs = dataset_cls_kwargs
        self.kwargs = kwargs
        self.stopped = False
        self.loaders = []

    def new_loader(self, info: dict):
        dataset = self.dataset_cls(
            path=self.input_file(info),
            preprocess=self.preprocess,
            **self.dataset_cls_kwargs,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            timeout=self.timeout,
            **self.kwargs,
        )
        return iter(dataloader), info

    def input_file(self, info: dict):
        return info["file"]

    def output_file(self, file: str):
        """specify output_file path for a file."""
        raise NotImplementedError()

    def should_skip(self, info: dict):
        output_file = self.output_file(info["file"])
        if not output_file:
            return True
        info["output_file"] = output_file
        return False

    @classmethod
    def __nap(cls):
        try:
            time.sleep(cls.sleep_seconds)
        except:
            pass

    def fetch(self):
        raise NotImplementedError()

    def __fetch(self):
        if len(self.loaders) >= self.prefetch_size:
            self.__nap()
            return
        try:
            loader = self.fetch()
        except:
            self.stop()
            raise
        if loader is None:
            self.stop()
            self.__nap()
            return
        self.loaders.append(loader)

    def next(self):
        while not (self.loaders or self.stopped):
            self.__nap()
        if self.loaders:
            return self.loaders.pop(0)
        return None

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            try:
                self.__fetch()
            except KeyboardInterrupt:
                pass


class FileLoaderLoader(LoaderLoader):
    def __init__(
        self,
        files: List[str],
        dataset_cls: Type[PathDataset],
        preprocess: Callable = identical,
        collate_fn: Callable = identical,
        batch_size=512,
        num_workers=5,
        timeout=10,
        prefetch_size=1,
        **kwargs,
    ):
        super().__init__(
            dataset_cls,
            preprocess,
            collate_fn,
            batch_size,
            num_workers,
            timeout,
            prefetch_size,
            **kwargs,
        )
        self.files = files
        self.idx = 0

    def fetch(self):
        while self.idx < len(self.files):
            info = {"file": self.files[self.idx]}
            self.idx += 1
            if self.should_skip(info):
                continue
            return self.new_loader(info)
        return None


class InfoLoaderLoader(LoaderLoader):
    def __init__(
        self,
        infos: List[dict],
        dataset_cls: Type[PathDataset],
        preprocess: Callable = identical,
        collate_fn: Callable = identical,
        batch_size=512,
        num_workers=5,
        timeout=10,
        prefetch_size=1,
        **kwargs,
    ):
        super().__init__(
            dataset_cls,
            preprocess,
            collate_fn,
            batch_size,
            num_workers,
            timeout,
            prefetch_size,
            **kwargs,
        )
        self.infos = infos
        self.idx = 0

    def fetch(self):
        while self.idx < len(self.infos):
            info = self.infos[self.idx]
            self.idx += 1
            if self.should_skip(info):
                continue
            return self.new_loader(info)
        return None


class InfoQueueLoaderLoader(LoaderLoader):
    def __init__(
        self,
        queue,
        dataset_cls: Type[PathDataset],
        preprocess: Callable = identical,
        collate_fn: Callable = identical,
        batch_size=512,
        num_workers=5,
        timeout=10,
        prefetch_size=1,
        **kwargs,
    ):
        super().__init__(
            dataset_cls,
            preprocess,
            collate_fn,
            batch_size,
            num_workers,
            timeout,
            prefetch_size,
            **kwargs,
        )
        self.queue = queue

    def queue_get(self):
        info = self.queue.get()
        assert isinstance(info, dict)
        return info

    def fetch(self):
        while True:
            info = self.queue_get()
            if self.should_skip(info):
                continue
            break
        return self.new_loader(info)


class KafkaLoaderLoader(LoaderLoader):
    def __init__(
        self,
        path: str,
        group_id: str,
        dataset_cls: Type[PathDataset],
        preprocess: Callable = identical,
        collate_fn: Callable = identical,
        batch_size=512,
        num_workers=5,
        timeout=10,
        prefetch_size=1,
        poll_timeout=-1,
        **kwargs,
    ):
        super().__init__(
            dataset_cls,
            preprocess,
            collate_fn,
            batch_size,
            num_workers,
            timeout,
            prefetch_size,
            **kwargs,
        )
        from app.common.kafka_ext import get_kafka_python_consumer

        self.consumer = get_kafka_python_consumer(path, group_id)
        self.poll_timeout = poll_timeout
        self.last_msg = None

    def commit(self):
        if self.last_msg is not None:
            self.consumer.commit(message=self.last_msg)
            self.last_msg = None

    def fetch(self):
        while True:
            msg = self.consumer.poll(self.poll_timeout)
            if msg is None:
                if self.poll_timeout > 0:
                    return None
                else:
                    self.__nap()
                    continue
            if msg.error():
                raise Exception(msg.error())
            info = json_loads(msg.value())
            if self.should_skip(info):
                continue
            return (self.new_loader(info), msg)

    def stop(self):
        super().stop()
        self.consumer.close()

    def next(self):
        self.commit()
        next_item = super().next()
        if next_item is None:
            return None
        loader, self.last_msg = next_item
        return loader
