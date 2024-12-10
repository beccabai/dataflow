

import os
import re
from app.ml.utils.volc_utils import get_user_name, get_ray_head_ip_from_dir
from app.ml.utils.local_fs import list_local_fs_files_recursively
from app.ml.utils.attribute import map_attr_file_name

USER_NAME = get_user_name()
RAY_INFO_DIR = f"/fs-computility/llm/shared/{USER_NAME}/ray_server_infos"


import ray
# always use 10001 as ray port
# set log_to_driver to False to avoid the log being too large
ray.init(f"ray://{get_ray_head_ip_from_dir(RAY_INFO_DIR)}:10001", log_to_driver = True)



@ray.remote(num_cpus=5, num_gpus=1, max_retries=5, retry_exceptions=True)
def process_files_by_model(records: list, model_config: dict, writer_config: dict=None):
    
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
                                  model.collate_fn)
    loaderloader.start()
    process_in_loop(model.inference, loaderloader,writer_config=writer_config)


model_config = {
    "model_class": "app.ml.model.BertTwoClassModel",
    "model_name": "ad_cluster_best_ckpt",
    "model_path": "/fs-computility/llm/shared/tanghuanze/0202_ad_bert/log-bert-base-uncased/ad_cluster_best_ckpt",
    "to_float16": True,
    "to_bettertransformer": True,
}




if __name__ == "__main__":
    # define input path this is a local file-system path
    input_path = "/fs-computility/llm/shared/qiujiantao/project/ray_inference/small_input"
    
    # list all jsonl files in input_path recursively
    files = list_local_fs_files_recursively(input_path)
    
    # define attribute key
    attr_key = None
    
    # map the input files to output files only replace the directory
    def map_dir(input_dir: str):
        raise NotImplementedError

    # map the input files to output files, both replace the directory and add attribute key
    def map_record(input_file: str, attr_key: str):
        input_dir = os.path.dirname(input_file)
        input_file = os.path.basename(input_file)
        output_dir = map_dir(input_dir)
        output_file = map_attr_file_name(input_file, attr_key)
        
        return {"file": os.path.join(input_dir,input_file), "output_file": os.path.join(output_dir, output_file)}

    # build a list of records to process
    ray_task_record_list =[map_record(f, attr_key) for f in files]
    
    def split_records(records, num_splits):
        """
        Split one record list into num_splits lists
        for example:
        records = [1,2,3,4,5,6,7,8,9,10]
        num_splits = 3
        return [[1,4,7,10], [2,5,8], [3,6,9]]
        
        Args:
            records: list of records
            num_splits: number of splits

        Returns:
            list of record lists
        """
        if len(records) < num_splits:
            return [[r] for r in records]
        else:
            return [records[i::num_splits] for i in range(num_splits)]
    
    TASK_SPLIT_NUM = 32 
    # the TASK_SPLIT_NUM is the max parallelism
    # if too small, the parallelism will be limited
    # if too large, the model will be loaded repeatedly too many times
    # best to be the number of gpus in the ray cluster 
    def post_process_func(doc_dict):
        return doc_dict["ad_cluster_best_ckpt_prob"]

    writer_config = {"mode": "scaler", "dtype": "float64", "post_process_func": post_process_func}
    ray_results = [process_files_by_model.remote(split, model_config, writer_config) for split in split_records(ray_task_record_list, TASK_SPLIT_NUM)]
    
    ray.get(ray_results)
