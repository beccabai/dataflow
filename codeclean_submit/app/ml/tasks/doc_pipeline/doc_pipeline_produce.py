import itertools
import re
from dataclasses import asdict, dataclass

import click
from loguru import logger

from app.common.json_util import json_dumps
from app.common.kafka_ext import get_kafka_python_producer
from app.common.s3 import get_s3_client, is_s3_object_exists, list_s3_objects

__re_s3_dir = re.compile("^s3://[^/]+/.{2,}/$")


@dataclass
class Msg:
    input_path: str
    output_path: str


def check_s3_dir(ctx, param, value):
    if not __re_s3_dir.match(value):
        raise click.BadParameter("s3_dir should be like s3://bucket/dir/")
    return value


@click.command()
@click.option(
    "--kafka_path",
    type=str,
    default="kafka://kafka_ali/doc_pipeline_test",
    help="kafka topic, e.g. kafka://kafka_ali/doc_pipeline_test",
)
@click.option("--input_dir", type=str, required=True, callback=check_s3_dir)
@click.option("--output_dir", type=str, required=True, callback=check_s3_dir)
def command(input_dir: str, output_dir: str, kafka_path: str):
    logger.info(f"{input_dir=} {output_dir=} {kafka_path=}")

    s3_client = get_s3_client(path=input_dir)
    for i, path in enumerate(itertools.islice(list_s3_objects(client=s3_client, path=input_dir, recursive=True), 10)):
        output_path = path.replace(input_dir, output_dir)
        msg = Msg(
            input_path=path,
            output_path=output_path,
        )

        logger.info(f"line: {i} | produce msg: {msg}")
    logger.info("Are you sure to produce this path msgs? [y/n]")
    if input() != "y":
        logger.info("exit")
        return

    p = get_kafka_python_producer(kafka_path)
    topic = kafka_path.split("/")[-1]

    counter = 0
    for path in list_s3_objects(client=s3_client, path=input_dir, recursive=True):
        output_path = path.replace(input_dir, output_dir)
        if not is_s3_object_exists(path):
            logger.warning(f"input path {path} not exists")
            continue

        if is_s3_object_exists(output_path):
            logger.info(f"output path {output_path} already exists")
            continue

        msg = Msg(
            input_path=path,
            output_path=output_path,
        )

        p.produce(topic, value=json_dumps(asdict(msg)))
        counter += 1
        if counter % 1000 == 0:
            p.poll(0)
            p.flush

            logger.info(f"line: {counter} | produce msg: {msg}")

    logger.info(f"total produce {counter} msgs")

    p.poll(0)
    p.flush()


if __name__ == "__main__":
    command()
