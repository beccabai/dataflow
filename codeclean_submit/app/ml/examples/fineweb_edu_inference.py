

import os
import re

# TODO make ray header management more elegant
RAY_INFO_DIR = "/fs-computility/llm/shared/maren/ray_inference/ray_server_infos/"

def get_ray_head_ip():
    """
    Get the ip of the ray head node
    by listing all dir in RAY_INFO_DIR and assert only one dir and the dir name is ip
    """
    def is_ip(input_str):
        return re.match(r"\d+\.\d+\.\d+\.\d+", input_str)

    # list all dir in RAY_INFO_DIR like "/fs-computility/llm/shared/qiujiantao/ray_server_infos/10.18.0.55_6379"
    dir_list = [f for f in os.listdir(RAY_INFO_DIR) if os.path.isdir(os.path.join(RAY_INFO_DIR, f)) and is_ip(f)]
    # assert only one dir and the dir name is ip
    assert len(dir_list) == 1
    return dir_list[0].split("_")[0]

   

import ray
# always use 10001 as ray port
ray.init(f"ray://{get_ray_head_ip()}:10001")

@ray.remote(num_gpus=1, max_retries=5, retry_exceptions=True)
def process_files_by_model(records: list, model_config: dict):
    
    import importlib
    model_class = model_config.pop("model_class")
    module_name, class_name = model_class.rsplit(".", 1)
    
    module = importlib.import_module(module_name)
    model = getattr(module, class_name)(model_config, "cuda")

    import os
    from app.ml.loader import InfoLoaderLoader
    from app.ml.dataset import JsonlDataset
    from app.ml.process import process_in_loop

    class MyLoaderLoader(InfoLoaderLoader):
        def should_skip(self, info: dict):
            input_file = info["file"]
            if not os.path.exists(input_file):
                return True
            
            output_file = info["output_file"]
            if os.path.exists(output_file):
                return True
            
            return False


    loaderloader = MyLoaderLoader(records,
                                  JsonlDataset,
                                  model.preprocess,
                                  model.collate_fn,
                                  batch_size=2048,
                                  num_workers=20,
                                  )
    loaderloader.start()
    process_in_loop(model.inference, loaderloader)


model_config = {
    "model_class": "app.ml.model.BertClassRegressionModel",
    "model_name": "fineweb-edu-classifier",
    "model_path": "/fs-computility/llm/shared/maren/models/finewebedu-classifier-official/",
    "to_float16": True,
    "to_bettertransformer": True,
    "use_logits": True,  # 直接输出regression head的值
    "use_clip": True, # 输出有范围限制
    "clip_min": 0, # 最小值
    "clip_max": 5 # 最大值
}

# ls /fs-computility/llm/shared/qiujiantao/project/ray_inference/test_input recuresively
def get_all_jsonl_files(input_dir: str):
    all_file_list = []
    import os
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                all_file_list.append(os.path.join(root, file))
    return all_file_list
        

# input_path = "/fs-computility/llm/shared/qiujiantao/project/ray_inference/test_input"
# input_path = "/fs-computility/llm/shared/llm_data/internlm_3_0825/en/CC-MAIN-2023-14"
input_path = "/fs-computility/llm/shared/llm_data/internlm_3_0825/en/"
output_path = "/fs-computility/llm/shared/maren/ray_inference/internlm_3_0825/en/"
os.makedirs(output_path, exist_ok=True)

def map_record(input_file: str):
    # output_file = input_file.replace("test_input", "test_output")
    output_file = input_file.replace("llm_data", "maren/ray_inference")
    return {"file": input_file, "output_file": output_file}
files =  get_all_jsonl_files(input_path)
input_file_list =[map_record(f) for f in files]

def split_records(records, num_splits):
    return [records[i::num_splits] for i in range(num_splits)]

res = [process_files_by_model.remote(split, model_config) for split in split_records(input_file_list, 200)]
ray.get(res)
