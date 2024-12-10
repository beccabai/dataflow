import sys
sys.path.append('/mnt/petrelfs/baitianyi/eval/Open-DataFlow-Eval/codeclean')
from datasets import load_dataset
import json
import pyarrow.parquet as pq
from dataflow.utils.registry import FORMATTER_REGISTRY
from dataflow.data.text_dataset import TextDataset
import os
from codeclean.app.ml.dataset import S3JsonlDataset 
import boto3 as boto
import json
from codeclean.app.common.s3 import *

@FORMATTER_REGISTRY.register()
class TextFormatter:
    def __init__(self, cfg):
        self.dataset_name = cfg.get('dataset_name', None) 
        self.dataset_split = cfg.get('dataset_split', None) 
        self.name = cfg.get('name', None) 
        self.data_dir = cfg.get('data_path', None) 
        self.keys = cfg.get('keys', None)  
        self.use_hf = cfg.get('use_hf')

    def load_dataset(self) -> TextDataset:
        if self.use_hf:
            return self.load_hf_dataset(
                dataset_name=self.dataset_name,
                dataset_split=self.dataset_split,
                name=self.name,
                keys=self.keys
            )
        elif self.data_dir:
            if self.data_dir.startswith("s3://"):
                return self.load_s3_dataset(self.data_dir)  
            else:
                return self.load_local_dataset(self.data_dir, self.keys) 
        else:
            raise RuntimeError("No valid dataset configuration found. Please provide either 'dataset_name' or 'data_dir'.")

    def load_hf_dataset(self, dataset_name, dataset_split=None, name=None, keys=None) -> TextDataset:
        load_kwargs = {
            "path": dataset_name,        
            "split": dataset_split,    
            "name": name                  
        }
        
        dataset = load_dataset(**{k: v for k, v in load_kwargs.items() if v is not None})

        metadata = {
            "description": dataset.info.description if hasattr(dataset, "info") else None,
            "features": dataset.info.features if hasattr(dataset, "info") else None,
            "version": dataset.info.version if hasattr(dataset, "info") else None
        }

        return TextDataset(
            dataset=dataset,
            keys=keys,
            metadata=metadata 
        )

    def load_s3_dataset(self, s3_path: str) -> TextDataset:
        print("Start loading....")
        s3_jsonl_dataset = S3JsonlDataset(s3_path, preprocess=lambda x:x)  
        print("Loading Done!!!")
        dataset = [item for item in s3_jsonl_dataset] 
        ####
        # dataset = dataset[:1000]
        ####
        return TextDataset(dataset=dataset, keys=self.keys)


    # def load_s3_dataset(self, s3_path: str) -> TextDataset:
    #     all_datasets = [] 
    #     metadata = [] 
    #     s3_client = get_s3_client(s3_path)
    #     files = list_s3_objects(s3_client, s3_path, recursive=True, is_prefix=False, limit=0)
    #     filtered_files = [file['Key'] for file in files if file['Key'].endswith('.jsonl.gz')]
    #     print(f"Files found: {filtered_files}")
    #     for file_path in filtered_files:
    #         s3_jsonl_dataset = S3JsonlDataset(file_path, preprocess=lambda x: x)
    #         for item in s3_jsonl_dataset:
    #             if "metadata" in item:
    #                 metadata.append(item.pop("metadata"))
    #             all_datasets.append(item)
    #     return TextDataset(
    #         dataset=all_datasets,  
    #         keys=self.keys,  
    #         metadata=metadata  
    #     )




    def load_local_dataset(self, file_path: str, keys=None) -> TextDataset:
        print("Start loading....")
        file_extension = os.path.splitext(file_path)[1].lower()
        metadata = None
        dataset = None

        # if file_extension == '.json':
        #     with open(file_path, 'r') as f:
        #         json_data = json.load(f)

        #     if "metadata" in json_data:
        #         metadata = json_data.pop("metadata")
            
        #     dataset = json_data["data"] if "data" in json_data else json_data
        
        # elif file_extension == '.jsonl':
        #     dataset = []
        #     metadata = None 

        #     with open(file_path, 'r') as f:
        #         for line in f:
        #             dataset.append(json.loads(line.strip()))
        
        # elif file_extension == '.parquet':
        #     table = pq.read_table(file_path)
        #     dataset = table.to_pydict()
        #     dataset = [{k: v[i] for k, v in dataset.items()} for i in range(len(next(iter(dataset.values()))))]
        #     metadata = table.schema.metadata  
        
        # else:
        #     raise RuntimeError(f"Unsupported file format: {file_extension}. Only .json, .jsonl and .parquet are supported.")
        
        all_datasets = []
        metadata = []
        # files_path = file_path
        def process_path(path):
            if os.path.isfile(path):
                process_file(path)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        process_file(file_path)

        def process_file(file_path):
            file_extension = os.path.splitext(file_path)[1].lower()
            dataset = None

            if file_extension == '.json':
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                if "metadata" in json_data:
                    metadata.append(json_data.pop("metadata"))
                
                dataset = json_data["data"] if "data" in json_data else json_data
            
            elif file_extension == '.jsonl':
                dataset = []
                with open(file_path, 'r') as f:
                    for line in f:
                        dataset.append(json.loads(line.strip()))
            
            elif file_extension == '.parquet':
                table = pq.read_table(file_path)
                parquet_data = table.to_pydict()
                dataset = [{k: v[i] for k, v in parquet_data.items()} for i in range(len(next(iter(parquet_data.values()))))]
                if table.schema.metadata:
                    metadata.append(table.schema.metadata)
            
            if dataset:
                all_datasets.extend(dataset)
                    
        process_path(file_path) 
        print("Loading Done!!!")            
        return TextDataset(
            dataset=all_datasets,
            # dataset=dataset,
            keys=keys,
            metadata=metadata 
        )
