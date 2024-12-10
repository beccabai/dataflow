def kafka_example():
    from app.common.s3 import head_s3_object_with_retry
    from app.ml.dataset import S3ImageH5Dataset
    from app.ml.img_cls import ImgClassifier
    from app.ml.loader import KafkaLoaderLoader
    from app.ml.process import process_in_loop

    class MyLoaderLoader(KafkaLoaderLoader):
        def should_skip(self, info: dict):
            input_file = info["file"]
            if "proc-zbar-h5" not in input_file:
                return True
            output_file = input_file.replace("proc-zbar-h5", "proc-zbar-h5-label") + ".jsonl"
            info["output_file"] = output_file
            if head_s3_object_with_retry(output_file):
                return True
            if not head_s3_object_with_retry(input_file):
                return True
            return False

    model = ImgClassifier()

    loader_loader = MyLoaderLoader(
        "kafka://kafka_3/jzj_test_files",
        "jzj_test",
        S3ImageH5Dataset,
        model.preprocess,
        model.collate_fn,
        poll_timeout=60,
    )
    loader_loader.start()

    process_in_loop(model.inference, loader_loader)


def ray_example():
    import ray

    from app.common.s3 import head_s3_object_with_retry
    from app.ml.loader import FileLoaderLoader

    class MyLoaderLoader(FileLoaderLoader):
        def should_skip(self, info: dict):
            input_file = info["file"]
            if "proc-zbar-h5" not in input_file:
                return True
            output_file = input_file.replace("proc-zbar-h5", "proc-zbar-h5-label") + ".jsonl"
            info["output_file"] = output_file
            if head_s3_object_with_retry(output_file):
                return True
            if not head_s3_object_with_retry(input_file):
                return True
            return False

    @ray.remote(num_gpus=1, max_retries=5, retry_exceptions=True)
    def process_files(files):
        import os

        os.chdir("/tmp")

        from app.ml.dataset import S3ImageH5Dataset
        from app.ml.img_cls import ImgClassifier
        from app.ml.process import process_in_loop

        model = ImgClassifier()

        loader_loader = MyLoaderLoader(
            files,
            S3ImageH5Dataset,
            model.preprocess,
            model.collate_fn,
        )
        loader_loader.start()
        process_in_loop(model.inference, loader_loader)

        return True

    file_splits = [["...", "...", "..."], ["...", "...", "..."]]
    refs = [process_files.remote(split) for split in file_splits]
    ray.get(refs)
