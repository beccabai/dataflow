import os
import json
import time
import sys
from importlib import import_module
from app.ml.utils.attribute import ContentAttributeReader
from app.ml.utils.local_fs import list_local_fs_files_recursively
from app.ml.utils.volc_utils import get_user_name, get_ray_head_ip_from_dir


USER_NAME = get_user_name()
RAY_INFO_DIR = f"/fs-computility/llm/shared/{USER_NAME}/ray_server_infos"


# set the path of the attribute filter ops
INTERNTRAIN_SF_BASE_DIR = os.environ.get("INTERNTRAIN_SF_BASE_DIR", None)
assert INTERNTRAIN_SF_BASE_DIR is not None, "Please set environment variable INTERNTRAIN_SF_BASE_DIR"
INTERNTRAIN_ATTRIBUTE_PATH = os.path.join(INTERNTRAIN_SF_BASE_DIR, "internlm/data/streaming/attribute")
assert os.path.exists(INTERNTRAIN_ATTRIBUTE_PATH), f"Path {INTERNTRAIN_ATTRIBUTE_PATH} does not exist"


import ray

# always use 10001 as ray port
# set log_to_driver to False to avoid the log being too large
ray.init(f"ray://{get_ray_head_ip_from_dir(RAY_INFO_DIR)}:10001", log_to_driver=True)


@ray.remote(num_cpus=1)
def static_file_token(sub_path: str, file_path: str, task_config: dict):
    """
    用于统计每个文件的筛选后token预测值
    先

    Args:
        sub_path (str): 用于确定哪个子集
        file_path (str): jsonl文件路径
        task_config (dict): 扫描任务配置，具体配置包括：
            content_root_dir ：content根目录
            attr_root_dir ： 属性根目录
            attr_key_list ： 需要扫描的属性list
            content_attr_reader_config ： CAR所需配置
            calc_token_interval ： 间隔多少有效行进行token数计算
            filter_func_str：用于筛选数据的函数名
            dynamic_func_str： 用于动态生成数据的函数名
            tokenizer_path： tokenizer的位置
            early_stop： 统计完多少条数据后结束


    Returns:
        result_dit :{
        "sub_path": sub_path, 用于确定哪个子集
        "file_path": file_path, jsonl文件路径
        "estimate_tokens": estimate_tokens, 估计的token数
        }
    """

    # load task config
    content_root_dir = task_config.get("content_root_dir", None)
    attr_root_dir = task_config.get("attr_root_dir", None)
    attr_key_list = task_config.get("attr_key_list", None)
    content_attr_reader_config = task_config.get("content_attr_reader_config", None)
    calc_token_interval = int(task_config.get("calc_token_interval", 1000))
    filter_func_str = task_config.get("filter_func_str", None)
    dynamic_func_str = task_config.get("dynamic_func_str", None)
    tokenizer_path = task_config.get("tokenizer_path", None)
    early_stop = task_config.get("early_stop", 0)

    # in this task we need to use line bytes to estimate total token in a file
    if content_attr_reader_config is None:
        content_attr_reader_config = {"calc_line_bytes": True}
    else:
        content_attr_reader_config["calc_line_bytes"] = True

    # force to use 1000 as calc_token_interval
    if calc_token_interval <= 0:
        calc_token_interval = 1000

    # load filter and dynamic functions
    sys.path.insert(0, INTERNTRAIN_ATTRIBUTE_PATH)

    if filter_func_str is not None:
        filter_drop_func = getattr(import_module("attibute_filter_ops"), filter_func_str)
    else:

        def dummy_fun_filter(**kwargs):
            return False

        filter_drop_func = dummy_fun_filter

    if dynamic_func_str is not None:
        dynamic_drop_func = getattr(import_module("attibute_filter_ops"), dynamic_func_str)
    else:

        def dummy_fun_dynamic(**kwargs):
            return {"content": kwargs["content"]}

        dynamic_drop_func = dummy_fun_dynamic

    # load tokenizer
    if calc_token_interval > 0:
        import sentencepiece as spm

        token_processor = spm.SentencePieceProcessor()
        token_processor.load(tokenizer_path)

        def calc_token(content: str) -> int:
            return len(token_processor.encode_as_ids(content))

    used_line_num = 0
    used_cbytes = 0

    sample_tokens = 0
    sample_cbytes = 0

    readed_line_bytes = 0
    readed_line_num = 0

    with ContentAttributeReader(
        file_path,
        content_root_dir,
        attr_root_dir,
        attr_key_list=attr_key_list,
        config=content_attr_reader_config,
    ) as content_attribute_reader:
        for data in content_attribute_reader:
            # apply filter and dynamic functions
            readed_line_num += 1
            readed_line_bytes += data["line_bytes"]
            try:
                drop_flag = filter_drop_func(**data)
            except Exception as e:
                raise Exception(f"KeyError: {e} in {file_path} by filter function {filter_func_str}")
            if drop_flag:
                continue

            try:
                new_content: str = dynamic_drop_func(**data)["content"]
            except Exception as e:
                raise Exception(f"KeyError: {e} in {file_path} by dynamic function {dynamic_func_str}")

            cbytes = len(new_content.encode("utf-8"))

            if sample_cbytes == 0 or (used_line_num % calc_token_interval == 0):
                token_num = calc_token(new_content)
                sample_cbytes += cbytes
                sample_tokens += token_num

            used_cbytes += cbytes
            used_line_num += 1
            if early_stop > 0 and used_line_num >= early_stop:
                break

    token_per_cbytes = sample_tokens / sample_cbytes
    cbytes_per_line = used_cbytes / used_line_num
    line_per_bytes = readed_line_num / readed_line_bytes
    filter_remained_rate = used_line_num/readed_line_num
    total_file_bytes = os.path.getsize(file_path)
    
    estimate_line_num = int(line_per_bytes * filter_remained_rate * total_file_bytes)
    estimate_cbytes = int(cbytes_per_line * line_per_bytes * filter_remained_rate * total_file_bytes)
    estimate_tokens = int(token_per_cbytes * cbytes_per_line * line_per_bytes * filter_remained_rate * total_file_bytes)

    return {
        "sub_path": sub_path,
        "file_path": file_path,
        "estimate_line_num": estimate_line_num,
        "estimate_cbytes": estimate_cbytes,
        "estimate_tokens": estimate_tokens,
        "token_per_cbytes": token_per_cbytes,
        "cbytes_per_line": cbytes_per_line,
        "line_per_bytes": line_per_bytes,
        "filter_remained_rate": filter_remained_rate,
        "total_file_bytes": total_file_bytes,
    }


def calc_dataset_weights(estimate_result_map: dict[str, dict[str, int]]):
    dataset_weights = {}
    sum_tokens = sum([sum(estimate_result_map[sub_path].values()) for sub_path in estimate_result_map.keys()])
    for sub_path in estimate_result_map.keys():
        sub_path_tokens = sum(estimate_result_map[sub_path].values())
        dataset_weights[sub_path] = sub_path_tokens / sum_tokens
    # sort key by weight
    new_dataset_weights = {key: dataset_weights[key] for key in sorted(dataset_weights.keys(), key=lambda x: dataset_weights[x], reverse=True)}
    return new_dataset_weights


def parse_config_py(exp_config_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", exp_config_py)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return vars(config)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config_py", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--early_stop", type=int, default=10000)
    args = parser.parse_args()

    exp_config_py = args.exp_config_py
    result_dir = args.result_dir
    early_stop = args.early_stop

    exp_config_dict = parse_config_py(exp_config_py)

    if result_dir is None:
        # if result_dir is None write result in same dir with exp_config_py
        result_dir = os.path.dirname(exp_config_py)

    config_py_name = os.path.basename(exp_config_py)
    if config_py_name == "data_config.py":
        config_prefix = ""
    else:
        config_prefix = config_py_name.replace(".py", "_")

    dataset_weights_path = os.path.join(result_dir, f"{config_prefix}dataset_weights.py")
    file_wise_result_path = os.path.join(result_dir, f"{config_prefix}param_filter_token_num.jsonl")

    train_folder = exp_config_dict["train_folder"]
    meta_folder = exp_config_dict["meta_folder"]
    vocab_file = exp_config_dict["vocab_file"]
    subset_params = exp_config_dict["subset_params"]

    subpath_files_map = {}

    def list_two_level_dir(root_dir):
        first_level = [x for x in os.listdir(root_dir)]
        second_level = {}
        for first_dir in first_level:
            for second_dir in [x for x in os.listdir(os.path.join(root_dir, first_dir))]:
                second_level[second_dir] = os.path.join(root_dir, first_dir, second_dir)
        return second_level

    two_level_dir = list_two_level_dir(train_folder)

    for subpath_path in subset_params.keys():
        assert subpath_path in two_level_dir.keys()
        full_path = os.path.join(train_folder, two_level_dir[subpath_path])
        jsonl_files = list_local_fs_files_recursively(full_path, ".jsonl")
        print(f"subpath: {subpath_path}, file number: {len(jsonl_files)}, full_path: {full_path}")
        subpath_files_map[subpath_path] = jsonl_files

    base_process_config = {
        "tokenizer_path": vocab_file,
        "calc_token_interval": 1e4,
        "filter_func": None,
        "dynamic_func_str": None,
        "content_root_dir": train_folder,
        "attr_root_dir": None,
        "attr_key_list": None,
        "early_stop": early_stop,
        "content_attr_reader_config": {"name_map": {"len": "_para_nums", "off": "_doc_offset", "flat": "_flat"}},
    }
    # build ray task list
    ray_task_list = []

    # each file will be processed by a ray task
    # result of each task is like {"sub_path": sub_path, "file_path": file_path, "estimate_tokens": estimate_tokens}
    # so we can gather the result of all tasks to get the total token number for each sub_path
    for sub_path in subpath_files_map.keys():
        jsonl_files = subpath_files_map[sub_path]
        for jsonl_file in jsonl_files:
            process_config = base_process_config.copy()
            process_config["attr_root_dir"] = subset_params[sub_path]["meta_folder"]
            process_config["attr_key_list"] = subset_params[sub_path]["attributes"]
            process_config["filter_func_str"] = subset_params[sub_path].get("prev_filter_func_str", None)
            process_config["dynamic_func_str"] = subset_params[sub_path].get("dynamic_func_str", None)
            ray_task_list.append(static_file_token.remote(sub_path=sub_path, file_path=jsonl_file, task_config=process_config))

    estimate_result_map = {}

    file_wise_result_fp = open(file_wise_result_path, "w")
    start_time = time.time()

    unready_list = ray_task_list

    # run the tasks in ray framework and wait for all tasks to finish
    print(f"start to process {len(ray_task_list)} tasks")
    while len(unready_list) > 0:
        print(f"waiting for {len(unready_list)} tasks")
        ready_list, unready_list = ray.wait(unready_list, num_returns=len(unready_list), timeout=1)

        now_time = time.time()
        print(
            f"finished tasks: {len(ray_task_list) - len(unready_list)}, unfinished tasks: {len(unready_list)}, time used:{now_time - start_time}"
        )

        for result in ready_list:
            line_result = ray.get(result)
            file_wise_result_fp.write(json.dumps(line_result) + "\n")
            file_wise_result_fp.flush()
            sub_path = line_result["sub_path"]
            file_path = line_result["file_path"]
            estimate_tokens = line_result["estimate_tokens"]
            if sub_path not in estimate_result_map:
                estimate_result_map[sub_path] = {}
            estimate_result_map[sub_path][file_path] = estimate_tokens

        if len(unready_list) == 0:
            break

    file_wise_result_fp.close()

    # 根据每个子集下的文件的token数来估计weight
    # 估计出来的weight以python脚本的形式输出，用于下一步进行config patch
    dataset_weights = calc_dataset_weights(estimate_result_map)
    import pprint

    start_str = "################ <DATACONFIG::dataset_weights> start ################"
    finish_str = "################ <DATACONFIG::dataset_weights> finish ################"
    dict_str = pprint.pformat(dataset_weights)
    dict_str = f"dataset_weights = {dict_str}"
    with open(dataset_weights_path, "w") as f:
        f.write("\n".join([start_str, dict_str, finish_str]))
