import sys
import os
import logging
import argparse
import ray
import re
import dataflow
import yaml
from dataflow.utils import calculate_score
import time

    
def setup_logging(log_file):
    with open(log_file, 'w'):
        pass
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    class LogToFile:
        def __init__(self):
            self.log_file = open(os.devnull, 'w')  

        def write(self, message):
            if message.strip():  #
                logging.info(message.strip())

        def flush(self):
            pass

        def fileno(self):
            return self.log_file.fileno()  
        
        def isatty(self):
            return False

    sys.stdout = LogToFile()
    sys.stderr = LogToFile()
    
def load_config_from_yaml(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def split_records(records, num_splits):
    if len(records) < num_splits:
        return [[r] for r in records]
    else:
        return [records[i::num_splits] for i in range(num_splits)]
    
def list_local_fs_files_recursively(file_path):
    with open(file_path, "r") as file:
        paths_list = [line.strip() for line in file if line.strip()]
    return paths_list
    
def process_files_with_ray(files_list, output_dir, cfg, num_splits):
    os.makedirs(output_dir, exist_ok=True)
    split_lists = split_records(files_list, num_splits)    
    results = []
    API_KEYS = [
    'AIzaSyCBmZuYLUpXjIvc0B73e6WG6dorXhzU7Kk', 'AIzaSyCK9teFVz_GYSxrsVYnJX42-HFEYDPpeJE',
    'AIzaSyBFt3rQrj001AsIrFniz8lqynt2CkqpMMA', 'AIzaSyC3kEyahFUw9hCgW3_M2ogz0B0TlBOoPeg',
    'AIzaSyCm4F0FjFU006fvrOSVemGaaZHBLvbVCI4', 'AIzaSyCg1PLPvJxsshmS1AMRYpjGR7VJou2ZYGU',
    'AIzaSyBcsMnmKB3okX4tjli881pyexkqraqsZjU', 'AIzaSyAXuiKRJ41XgTngmR9YTEKpbLRBChUu5Is',
    'AIzaSyCBeRUdEmY4jK3DtR16P1ECA3Y4r5s0PN8', 'AIzaSyC_wKcLvXWptGbsQqnU-xxm04i_fAbRrnA'
    ]

    for split_idx, split in enumerate(split_lists):
        if "PerspectiveScorer" in cfg["scorers"]:
            cfg["scorers"]["PerspectiveScorer"]["api_key"] = API_KEYS[split_idx]
        print(f"Submitting split {split_idx} with {len(split)} files.")
        results.append(calculate_score.remote(split, output_dir, cfg))

    ray.get(results)
    print("Finished processing all files.")
    
def main():
    parser = argparse.ArgumentParser(description="Run score calculation.")
    parser.add_argument("--output", help="Path to save the output results JSON file.")
    parser.add_argument("--log", help="Path to save the log file.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset.")
    parser.add_argument("--scorer", required=True, nargs='+', help="Name of the scorer.")
    args = parser.parse_args()
    
    output_dir = f'/mnt/petrelfs/baitianyi/eval/Open-DataFlow-Eval/eval_output_sft/{args.dataset}/{args.scorer[0]}'
    log_dir = f'/mnt/petrelfs/baitianyi/eval/Open-DataFlow-Eval/eval_log_sft/{args.dataset}'
    log_file = f'{args.scorer[0]}.log'

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_file)
    setup_logging(log_file)
    
    try:
        dataflow_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        sys.path.insert(0, dataflow_path)
        sys.argv = [
            'notebook',
            '--config',
            '/mnt/petrelfs/baitianyi/eval/Open-DataFlow-Eval/configs/text_scorer_pt.yaml'
        ]
        
        file_path = f"/mnt/petrelfs/baitianyi/eval/Open-DataFlow-Eval/dataflow/Eval/sft/{args.dataset}.txt"
        # file_path = f"/mnt/petrelfs/baitianyi/eval/Open-DataFlow-Eval/dataflow/Eval/s3data/{args.dataset}.txt"
        cfg_path = "/mnt/petrelfs/baitianyi/eval/Open-DataFlow-Eval/configs/text_scorer_pt.yaml"
        TASK_SPLIT_NUM = 10
        paths_list = list_local_fs_files_recursively(file_path)
        
        logging.info("Reading config file...")
        cfg = load_config_from_yaml(cfg_path)
        cfg["scorers"] = {scorer: cfg["scorers"].get(scorer, {}) for scorer in args.scorer}
        
        logging.info("Starting score calculation...")
        process_files_with_ray(paths_list, output_dir, cfg, TASK_SPLIT_NUM)
        ray.shutdown()
    except Exception as e:
        logging.error(f"An error occurred during score calculation: {e}")

if __name__ == "__main__":
    ray.init()
    main()