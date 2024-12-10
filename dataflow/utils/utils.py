import numpy as np
import json
import subprocess
import numpy as np
import os
import ray
import yaml

def download_model_from_hf(model_name, model_cache_dir):
    print(f"Downloading {model_name} to {model_cache_dir}.")
    command = ['huggingface-cli', 'download', '--resume-download', model_name, '--local-dir', model_cache_dir]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to download {model_name}.")
        print(result.stderr)
        return False
    print(f"Successfully downloaded {model_name} to {model_cache_dir}.")
    return True

def round_to_sigfigs(num, sigfigs):
    import math
    if isinstance(num, np.float32):
        num = float(num)
    if num == 0:
        return 0
    else:
        return round(num, sigfigs - int(math.floor(math.log10(abs(num)))) - 1)


def recursive_insert(ds_scores_dict, scores: dict, idx_list):
    import torch
    import numpy as np
    for k, v in scores.items():
        if isinstance(v, dict):
            recursive_insert(ds_scores_dict[k], v, idx_list)
        elif isinstance(v, torch.Tensor):
            ds_scores_dict[k][idx_list] = v.cpu().detach().numpy()
        elif isinstance(v, list):
            ds_scores_dict[k][idx_list] = np.array(v)
        elif isinstance(v, float):
            ds_scores_dict[k][idx_list] = np.array(v)
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")

def recursive_func(scores: dict, func, output: dict):
    import numpy as np
    
    for k, v in scores.items():
        if isinstance(v, dict):
            if k not in output.keys():
                output[k] = {}
            recursive_func(scores[k], func, output[k])
        elif isinstance(v, (np.float64, np.float32, np.ndarray)):
            if isinstance(v, np.ndarray) and np.isnan(v).all():
                output[k] = v
            elif isinstance(v, (np.float64, np.float32)) and np.isnan(v):
                output[k] = v
            else:
                output[k] = func(v)
        elif isinstance(v, str):
            output[k] = v  
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")




def recursive_len(scores: dict):
    import numpy as np
    for _, v in scores.items():
        if isinstance(v, dict):
            return recursive_len(v)
        elif isinstance(v, np.ndarray):
            return v.shape[0]
        elif isinstance(v, list): 
            return len(v)
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")
        
def recursive_idx(scores: dict, index, output: dict):
    for k, v in scores.items():
        if isinstance(v, dict):
            if k not in output.keys():
                output[k] = {}
            recursive_idx(scores[k], index, output[k])
        elif isinstance(v, np.ndarray):
            output[k] = v[index]
        elif isinstance(v, list): 
            output[k] = v[index] 
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")

def recursive(scores: dict, output: dict):
    for k, v in scores.items():
        if isinstance(v, dict):
            if k not in output.keys():
                output[k] = {}
            recursive(scores[k], output[k])
        else:
            output[k] = v

def list_image_eval_metrics():
    from dataflow.config import init_config
    import pyiqa

    cfg = init_config()
    metric_dict = {}
    metric_dict['image']=pyiqa.list_models(metric_mode="NR")

    for k, v in cfg.image.items():
        if v['data_type'] in metric_dict:
            metric_dict[v['data_type']].append(k)
        else:
            metric_dict[v['data_type']] = [k]
    for k, v in metric_dict.items():
        print(f"metric for {k} data:")
        print(v)


def get_scorer(metric_name, device):
    from dataflow.config import init_config
    from dataflow.utils.registry import MODEL_REGISTRY
    import pyiqa

    cfg = init_config()
    if metric_name in cfg.image:
        model_args = cfg.image[metric_name]
        model_args['model_cache_dir'] = cfg.model_cache_path
        model_args['num_workers'] = cfg.num_workers
        scorer = MODEL_REGISTRY.get(model_args['class_name'])(device=device, args_dict=model_args)
    elif metric_name in pyiqa.list_models(metric_mode="NR"):
        # model_args={}
        model_args = cfg.image['pyiqa']
        model_args['model_cache_dir'] = cfg.model_cache_path
        model_args['num_workers'] = cfg.num_workers
        scorer = MODEL_REGISTRY.get(model_args['class_name'])(device=device, metric_name=metric_name, args_dict=model_args)
    elif metric_name in cfg.video:
        model_args = cfg.video[metric_name]
        scorer = MODEL_REGISTRY.get(metric_name)(model_args)
    else:
        raise ValueError(f"Metric {metric_name} is not supported.")
    
    assert scorer is not None, f"Scorer for {metric_name} is not found."
    return scorer

def new_get_scorer(scorer_name, model_args):
    from dataflow.utils.registry import MODEL_REGISTRY
    print(scorer_name, model_args)
    scorer = MODEL_REGISTRY.get(scorer_name)(args_dict=model_args)
    assert scorer is not None, f"Scorer for {scorer_name} is not found."
    return scorer


def map_output_path(input_path, output_dir):
    """
    Map an input file path to its corresponding output path.
    Args:
        input_path: str, input file path
        output_dir: str, directory for output files
    Returns:
        str, output file path
    """
    base_name = os.path.basename(input_path)  # Get the file name
    output_file = os.path.join(output_dir, base_name.replace(".jsonl.gz", "_output.json"))
    return output_file

@ray.remote(num_cpus=16, num_gpus=1, max_retries=5, retry_exceptions=True)
def calculate_score(paths_list= [], output_dir = None, cfg = None):
    from ..config import new_init_config
    from dataflow.utils.registry import FORMATTER_REGISTRY
    from dataflow.core import ScoreRecord

    # cfg = load_config_from_yaml(cfg_path)
    # cfg["scorers"] = {scorer: cfg["scorers"].get(scorer, {}) for scorer in scorers_to_process}
    
    for file_path in paths_list:
        output_file = map_output_path(file_path, output_dir)
        
        # Skip processing if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {file_path} because the output file already exists.")
            continue
        
        print(f"Processing {file_path} and saving to {output_file}") 
        dataset_dict = {}
        score_record = ScoreRecord()        
        for scorer_name, model_args in cfg["scorers"].items():
            if "num_workers" in cfg:
                model_args["num_workers"] = cfg["num_workers"]
            if "model_cache_path" in cfg:
                model_args["model_cache_dir"] = cfg["model_cache_path"]
            os.environ['http_proxy'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
            os.environ['https_proxy'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
            os.environ['HTTP_PROXY'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
            os.environ['HTTPS_PROXY'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
            scorer = new_get_scorer(scorer_name, model_args)
            cfg['data'][scorer.data_type]['data_path'] = file_path
            print(f"Processing {file_path}")
            if scorer.data_type not in dataset_dict:
                formatter = FORMATTER_REGISTRY.get(cfg['data'][scorer.data_type]['formatter'])(cfg['data'][scorer.data_type])
                os.environ.pop("http_proxy", None)
                os.environ.pop("https_proxy", None)
                os.environ.pop("HTTP_PROXY", None)
                os.environ.pop("HTTPS_PROXY", None)
                datasets = formatter.load_dataset()
                os.environ['http_proxy'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
                os.environ['https_proxy'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
                os.environ['HTTP_PROXY'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
                os.environ['HTTPS_PROXY'] = 'http://baitianyi:TwYpCc0xCLDfvXwaAFlEX8MXJeKw2RXUf6RpREQ16qC7vaopXs4zfl4xCmGh@10.1.20.51:23128/'
                dataset_dict[scorer.data_type] = datasets
                dataset = datasets[0] if type(datasets) == tuple else datasets
                dataset.set_score_record(score_record)
            else:
                datasets = dataset_dict[scorer.data_type]
            print("scoring datasets")
            
            _, score = scorer(datasets)        

        score_record.dump_scores(output_file)