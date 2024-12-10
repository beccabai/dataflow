

import os
import re
from app.ml.utils.volc_utils import get_user_name, get_ray_head_ip_from_dir
from app.ml.utils.local_fs import list_local_fs_files_recursively
from app.ml.utils.attribute import LocalAttributeWriter
USER_NAME = get_user_name()
RAY_INFO_DIR = f"/fs-computility/llm/shared/{USER_NAME}/ray_server_infos"


import ray
# always use 10001 as ray port
# set log_to_driver to False to avoid the log being too large
ray.init(f"ray://{get_ray_head_ip_from_dir(RAY_INFO_DIR)}:10001", log_to_driver = True)



@ray.remote(num_cpus=1, max_retries=5, retry_exceptions=True)
def process_files(records: list):
    import numpy as np
    for record in records:
        input_file = record["file"]
        output_file = record["output_file"]
        if os.path.exists(output_file):
            continue
        if not os.path.exists(input_file):
            continue
        attr_list = np.load(input_file, allow_pickle=True)["attr"]
        def get_line(line):
            return line
        writer = LocalAttributeWriter(output_file, get_line, dtype="int32", mode="flat2d")
        for attr in attr_list:
            writer.write(attr)
        writer.flush()
        print(f"write {output_file} done")
    
    






if __name__ == "__main__":
    # define input path this is a local file-system path
    input_path = "/fs-computility/llm/shared/qiujiantao/project/ray_inference/npz_input/"
    
    # list all jsonl files in input_path recursively
    files = list_local_fs_files_recursively(input_path, postfix=".npz")
    
    # map the input files to output files only replace the directory
    def map_dir(input_dir: str):
        return input_dir.replace("npz_input", "npy_output")

    # map the input files to output files, both replace the directory and add attribute key
    def map_record(input_file: str):
        input_dir = os.path.dirname(input_file)
        input_file = os.path.basename(input_file)
        output_dir = map_dir(input_dir)
        output_file = input_file.replace(".npz", "<flat2d>.2d.mmap.npy")
        
        return {"file": os.path.join(input_dir,input_file), "output_file": os.path.join(output_dir, output_file)}

    # build a list of records to process
    ray_task_record_list =[map_record(f) for f in files]
    print(f"total {ray_task_record_list} files to process")
    
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

    ray_results = [process_files.remote(split) for split in split_records(ray_task_record_list, TASK_SPLIT_NUM)]
    
    ray.get(ray_results)
