from tqdm import tqdm
import numpy as np
import torch
import json
from dataflow.data import DataFlowDataset, ImageCaptionDataset, ImageDataset, PureVideoDataset, TextDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataflow.utils import recursive_len, recursive_idx, recursive_insert, recursive_func, round_to_sigfigs, recursive
from functools import partial
import ray
import math
import os

class ScoreRecord():
    
    def __init__(self):
        self.item_score = {}
        self.meta_score = {}
                 
    def __getitem__(self, idx):
        val = {}
        recursive_idx(self.item_score, idx, val)
        rounded_val = {}
        recursive_func(val, partial(round_to_sigfigs, sigfigs=4), rounded_val)
        return rounded_val

    def dump_scores(self, filename=None):
        scores_len = recursive_len(self.item_score) if self.item_score else 0
        print(f"scores_len:{scores_len}")
        val = {}
        recursive(self.meta_score, val)
        rounded_meta_score = {}
        recursive_func(val, partial(round_to_sigfigs, sigfigs=4), rounded_meta_score)
        item_scores_indexed_and_rounded = {}
        for idx in range(scores_len):
            item_scores_indexed_and_rounded[str(idx)] = self[idx]
        if filename is None:
            print({
                    'meta_scores': rounded_meta_score,
                    'item_scores': item_scores_indexed_and_rounded,
                })
        else:
            with open(filename, 'w+') as f:
                json.dump({
                    'meta_scores': rounded_meta_score,
                    'item_scores': item_scores_indexed_and_rounded,
                }, f, indent=4)

    def calculate_statistics(self, scorer_name, score_name='Default'):
        data = self.item_score[scorer_name][score_name]
        num = len(data)
        max_value = np.max(data)
        min_value = np.min(data)
        mean_value = np.mean(data)
        variance_value = np.var(data)
        
        return {
            'num': num,
            'max': max_value,
            'min': min_value,
            'mean': mean_value,
            'variance': variance_value
        }


class Scorer:
    
    def __init__(self):
        self.scorer_name = ''
        pass

    def evaluate_batch(self, sample):
        raise NotImplementedError

    def evaluate(self, dataset: DataFlowDataset):
        raise NotImplementedError
        
    def __call__(self, dataset: DataFlowDataset):
        return self.evaluate(dataset)

class TextScorer(Scorer):
    def __init__(self, args_dict: dict):
        super().__init__()
        self.batch_size = args_dict.get('batch_size')
        self.num_workers = args_dict.get('num_workers', 0)
        # self.num_splits = args_dict.get('num_splits', 4)
        # self.num_cpus_per_task = args_dict.get('num_cpus_per_task', 8)
        # self.num_gpus_per_task = args_dict.get('num_gpus_per_task', 1)

    def init_score(self, len_dataset, dtype=float):
        if dtype == float:
            return {'Default': np.full(len_dataset, np.nan)} 
        elif dtype == str:
            return {'Default': [''] * len_dataset} 
        else:
            raise ValueError("Unsupported dtype for init_score")
        
    def evaluate_batch(self, batch):
        raise NotImplementedError
    
    

    # def create_split_dataloaders(self, dataset, batch_size, num_splits):
    #     """Create num_splits DataLoaders"""
    #     split_size = len(dataset) // num_splits
    #     remainder = len(dataset) % num_splits
    #     split_indices = [list(range(i * split_size + min(i, remainder), (i + 1) * split_size + min(i + 1, remainder))) for i in range(num_splits)]
        
    #     split_dataloaders = []
    #     for split in split_indices:
    #         sampler = SubsetRandomSampler(split)
    #         split_dataloaders.append(DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=self.num_workers))        
        
    #     return split_dataloaders
        
    # def evaluate(self, dataset: TextDataset):
    #     if self.batch_size == -1:
    #         self.batch_size = len(dataset)

    #     if not getattr(self, 'use_meta', False) and self.score_name not in dataset.score_record.item_score:
    #         dataset.score_record.item_score[self.score_name] = self.init_score(len(dataset), self.score_type)

    #     split_dataloaders = self.create_split_dataloaders(dataset, batch_size=self.batch_size, num_splits=self.num_splits)
    #     ray.init()

    #     # Define the remote function for evaluating a split
    #     @ray.remote(num_cpus=self.num_cpus_per_task, num_gpus=self.num_gpus_per_task, max_retries=1, retry_exceptions=True)
    #     def evaluate_split_remote(evaluate_batch_fn, dataloader, split_idx, score_name, dataset_keys):
    #         """Evaluate batch data using a remote function with minimal capture scope."""
    #         results = []

    #         for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating split {split_idx} - {score_name}", total=len(dataloader))):
    #             if isinstance(dataset_keys, list):
    #                 data_to_score = {key: batch[key] for key in dataset_keys}
    #             else:
    #                 data_to_score = {dataset_keys: batch[dataset_keys]}
    #             results.append(evaluate_batch_fn(data_to_score))

    #         return results

    #     # Store the dataloaders in Ray's object store to avoid passing large objects directly
    #     dataloader_refs = [ray.put(dataloader) for dataloader in split_dataloaders]

    #     results = []
    #     for split_idx, dataloader_ref in enumerate(dataloader_refs):
    #         print(f"Submitting split {split_idx} to remote execution")

    #         # Submit the remote task with dataloader reference
    #         result = evaluate_split_remote.remote(
    #             self.evaluate_batch, dataloader_ref, split_idx, self.score_name, dataset.keys
    #         )
    #         results.append((split_idx, result))
        
    #     split_size = len(dataset) // self.num_splits
    #     remainder = len(dataset) % self.num_splits
        
    #     for split_idx, remote_result in results:
    #         scores_list = ray.get(remote_result)
            
    #         split_start = split_idx * split_size + min(split_idx, remainder)
    #         split_end = split_start + split_size + (1 if split_idx < remainder else 0)

    #         for batch_idx, scores in enumerate(scores_list):
    #             idx_start = split_start + batch_idx * self.batch_size
    #             idx_end = min(idx_start + self.batch_size, split_end)
    #             idx_list = list(range(idx_start, idx_end))
                
    #             if getattr(self, 'use_meta', False):
    #                 dataset.score_record.meta_score[self.score_name] = scores
    #                 continue

    #             if isinstance(scores, dict):
    #                 for k, v in scores.items():
    #                     if k not in dataset.score_record.item_score[self.score_name]:
    #                         dataset.score_record.item_score[self.score_name][k] = np.full(len(dataset), np.nan)

    #                     if isinstance(v, torch.Tensor):
    #                         v = v.cpu().detach().numpy()

    #                     for i, idx_val in enumerate(idx_list):
    #                         dataset.score_record.item_score[self.score_name][k][idx_val] = v[i]

    #                 if 'Default' in dataset.score_record.item_score[self.score_name]:
    #                     default_scores = dataset.score_record.item_score[self.score_name]['Default']
    #                     if np.isnan(default_scores).all():
    #                         del dataset.score_record.item_score[self.score_name]['Default']

    #             elif isinstance(scores, list) or isinstance(scores, np.ndarray):
    #                 for i, idx_val in enumerate(idx_list):
    #                     dataset.score_record.item_score[self.score_name]['Default'][idx_val] = scores[i]

    #             elif isinstance(scores, torch.Tensor):
    #                 scores = scores.cpu().detach().numpy()
    #                 for i, idx_val in enumerate(idx_list):
    #                     dataset.score_record.item_score[self.score_name]['Default'][idx_val] = scores[i]
    #             else:
    #                 raise ValueError(f"Invalid scores type {type(scores)} returned by {self.score_name}")

    #     ray.shutdown()

    #     if getattr(self, 'use_meta', False):
    #         return self.scorer_name, dataset.score_record.meta_score[self.score_name]
    #     else:
    #         return self.scorer_name, dataset.score_record.item_score[self.score_name]
    
    def evaluate(self, dataset: TextDataset):
        if self.batch_size == -1:
            self.batch_size = len(dataset)

        if not getattr(self, 'use_meta', False) and self.score_name not in dataset.score_record.item_score:
            dataset.score_record.item_score[self.score_name] = self.init_score(len(dataset), self.score_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {self.score_name}", total=len(dataloader))):
            if isinstance(dataset.keys, list):
                data_to_score = {key: batch[key] for key in dataset.keys}
            else:
                data_to_score = {dataset.keys: batch[dataset.keys]}

            scores = self.evaluate_batch(data_to_score)

            if getattr(self, 'use_meta', False):
                dataset.score_record.meta_score[self.score_name] = scores 
                continue

            idx_list = list(range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(dataset))))

            if isinstance(scores, dict):
                for k, v in scores.items():
                    if k not in dataset.score_record.item_score[self.score_name]:
                        dataset.score_record.item_score[self.score_name][k] = np.full(len(dataset), np.nan)

                    if isinstance(v, torch.Tensor):
                        v = v.cpu().detach().numpy()

                    for i, idx_val in enumerate(idx_list):
                        dataset.score_record.item_score[self.score_name][k][idx_val] = v[i]

                if 'Default' in dataset.score_record.item_score[self.score_name]:
                    default_scores = dataset.score_record.item_score[self.score_name]['Default']
                    if np.isnan(default_scores).all():
                        del dataset.score_record.item_score[self.score_name]['Default']

            elif isinstance(scores, list) or isinstance(scores, np.ndarray):
                for i, idx_val in enumerate(idx_list):
                    dataset.score_record.item_score[self.score_name]['Default'][idx_val] = scores[i]
            elif isinstance(scores, torch.Tensor):
                scores = scores.cpu().detach().numpy()
                for i, idx_val in enumerate(idx_list):
                    dataset.score_record.item_score[self.score_name]['Default'][idx_val] = scores[i]
            else:
                raise ValueError(f"Invalid scores type {type(scores)} returned by {self.score_name}")
        if getattr(self, 'use_meta', False):
            return self.scorer_name, dataset.score_record.meta_score[self.score_name]
        else:
            return self.scorer_name, dataset.score_record.item_score[self.score_name]


class ImageScorer(Scorer):
    def __init__(self, args_dict: dict):
        super().__init__()
        # if not os.path.exists(args_dict['model_cache_dir']):
        #     os.makedirs(args_dict['model_csache_dir'])
        self.batch_size = args_dict['batch_size']
        self.num_workers = args_dict['num_workers']

    def init_score(self, len_dataset):
        return {'Default': np.array([np.nan] * len_dataset)}

    def evaluate_batch(self, sample):
        raise NotImplementedError

    def evaluate(self, dataset: ImageDataset):
        if not isinstance(dataset, ImageDataset):
            raise ValueError(f"The dataset should be an instance of ImageDataset, but got {type(dataset)}.")
        
        if hasattr(self, 'image_preprocessor'):
            dataset.set_image_preprocess(self.image_preprocessor)

        if hasattr(self, 'collate_fn'):
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        score_list = np.array([])
        # id_list = np.array([])

        for data in dataloader:
            scores = self.evaluate_batch(data[1])
            scores = scores.squeeze()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().detach().numpy()
            if len(scores.shape) == 0:
                scores = np.array([scores])
            score_list = np.concatenate([score_list, scores])
            # id_list = np.concatenate([id_list, data[0]])

        if self.scorer_name not in dataset.score_record.item_score:
            dataset.score_record.item_score[self.scorer_name] = self.init_score(len(dataset))
        idx_list = list(range(len(dataset)))
        dataset.score_record.item_score[self.scorer_name]['Default'][idx_list] = score_list
        # assert len(score_list) == len(id_list), "The number of scores and ids should be the same."
        # id_score_dict = dict(zip(id_list, score_list))
        # return id_score_dict

        # dataset.scores_list[self.scorer_name] = score_list
        # return score_list
        return self.scorer_name, dataset.score_record.item_score[self.scorer_name]
    
    def __call__(self, dataset: ImageDataset):
        return self.evaluate(dataset)
    

class ImageTextScorer(Scorer):
    def __init__(self, args_dict: dict):
        super().__init__()
        # if not os.path.exists(args_dict['model_cache_dir']):
        #     os.makedirs(args_dict['model_csache_dir'])
        self.batch_size = args_dict['batch_size']
        self.num_workers = args_dict['num_workers']

    def init_score(self, len_dataset):
        return {'Default': np.array([np.nan] * len_dataset)}

    def evaluate_batch(self, sample):
        raise NotImplementedError

    def evaluate(self, dataset: ImageCaptionDataset):
        if not isinstance(dataset, ImageCaptionDataset):
            raise ValueError(f"The dataset should be an instance of ImageTextDataset, but got {type(dataset)}.")

        if hasattr(self, 'image_preprocessor'):
            dataset.set_image_preprocess(self.image_preprocessor)
        if hasattr(self, 'text_preprocessor'):
            dataset.set_text_preprocess(self.text_preprocessor)

        if hasattr(self, 'collate_fn'):
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        score_list = np.array([])
        # id_list = np.array([])

        for data in dataloader:
            scores = self.evaluate_batch(data[1:])
            scores = scores.squeeze()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().detach().numpy()
            if len(scores.shape) == 0:
                scores = np.array([scores])
            score_list = np.concatenate([score_list, scores])
            # id_list = np.concatenate([id_list, data[0]])

        if self.scorer_name not in dataset.score_record.item_score:
            dataset.score_record.item_score[self.scorer_name] = self.init_score(len(dataset))
        idx_list = list(range(len(dataset)))
        dataset.score_record.item_score[self.scorer_name]['Default'][idx_list] = score_list

        # assert len(score_list) == len(id_list), "The number of scores and ids should be the same."
        # id_score_dict = dict(zip(id_list, score_list))
        # return id_score_dict
        # dataset.scores_list[self.scorer_name] = score_list
        return self.scorer_name, dataset.score_record.item_score[self.scorer_name]
    
    def __call__(self, dataset: ImageCaptionDataset):
        return self.evaluate(dataset)


class GenImageScorer(Scorer):
    def __init__(self, args_dict: dict):
        super().__init__()
        self.batch_size = args_dict['batch_size']
        self.num_workers = args_dict['num_workers']

    def init_score(self):
        '''
        return empty score for this scorer
        eg: {'Default': np.array(np.nan)}
        '''
        return {'Default': np.nan}

    def evaluate_batch(self, sample, ref_sample=None):
        raise NotImplementedError

    def evaluate(self, datasets):
        dataset = datasets[0]
        ref_dataset = datasets[1] if len(datasets) == 2 else None
        if self.scorer_name not in dataset.score_record.meta_score:
            dataset.score_record.meta_score[self.scorer_name] = self.init_score()
        if hasattr(self, 'image_preprocessor'):
            dataset.set_image_preprocess(self.image_preprocessor)

        if hasattr(self, 'collate_fn'):
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=self.num_workers)

        score_list = []
        if ref_dataset is None:
            for data in dataloader:
                scores = self.evaluate_batch(data[1])
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().detach().numpy()
                score_list.append(scores)
        else:
            if hasattr(self, 'image_preprocessor'):
                ref_dataset.set_image_preprocess(self.image_preprocessor)
            if hasattr(self, 'collate_fn'):
                ref_dataloader = DataLoader(ref_dataset, batch_size=len(ref_dataset), shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
            else:
                ref_dataloader = DataLoader(ref_dataset, batch_size=len(ref_dataset), shuffle=False, num_workers=self.num_workers)
            for data, ref_data in zip(dataloader, ref_dataloader):
                scores = self.evaluate_batch(data[1], ref_data[1])
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().detach().numpy
                score_list.append(scores) # list
                
        dataset.score_record.meta_score[self.scorer_name]['Default'] = score_list[0]
        # if ref_dataset is not None:
        #     ref_dataset.meta_score[self.scorer_name] = score_list[0]
        return self.scorer_name, dataset.score_record.meta_score[self.scorer_name]
        
class VideoScorer(Scorer):
    def __init__(self, args_dict: dict):
        super().__init__()
        self.batch_size = args_dict['batch_size']
        self.num_workers = args_dict['num_workers']
        self.data_type = 'video'
    
    def init_score(self, len_dataset):
        '''
        return empty score dict for this scorer
        eg: {'Default': np.array([-1] * len_dataset)}
        TODO: 
        '''
        return {'Default': np.array([np.nan] * len_dataset)}
    
    def evaluate_batch(self, sample):
        raise NotImplementedError

    def evaluate(self, dataset: PureVideoDataset):
        if self.scorer_name not in dataset.score_record.item_score:
            dataset.score_record.item_score[self.scorer_name] = self.init_score(len(dataset))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for idx, sample in enumerate(dataloader):
            scores = self.evaluate_batch(sample)
            idx_list = list(range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(dataset))))
            if isinstance(scores, dict):
                recursive_insert(dataset.score_record.item_score[self.scorer_name], scores, idx_list)
            elif isinstance(scores, list):
                    dataset.score_record.item_score[self.scorer_name]['Default'][idx_list] = np.array(scores)
            elif isinstance(scores, torch.Tensor):
                    dataset.score_record.item_score[self.scorer_name]['Default'][idx_list] = scores.cpu().detach().numpy()
            else:
                raise ValueError(f"Invalid scores type {type(scores)} returned by {self.scorer_name}")
            
        return self.scorer_name, dataset.score_record.item_score[self.scorer_name]

class VideoTextScorer(Scorer):
    def __init__(self, args_dict: dict):
        super().__init__()
        self.batch_size = args_dict['batch_size']
        self.num_workers = args_dict['num_workers']
        self.data_type = 'video-caption'

    def init_score(self, len_dataset):
        return {'Default': np.array([np.nan] * len_dataset)}
    
    def evaluate_batch(self, sample):
        raise NotImplementedError           
    
    def evaluate(self, dataset):
        if self.scorer_name not in dataset.score_record.item_score:
            dataset.score_record.item_score[self.scorer_name] = self.init_score(len(dataset))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for idx, sample in enumerate(dataloader):
            scores = self.evaluate_batch(sample)
            if scores is None:
                continue
            idx_list = list(range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(dataset))))
            if isinstance(scores, dict):
                recursive_insert(dataset.score_record.item_score[self.scorer_name], scores, idx_list)
            elif isinstance(scores, list):
                    dataset.score_record.item_score[self.scorer_name]['Default'][idx_list] = np.array(scores)
            elif isinstance(scores, torch.Tensor):
                    dataset.score_record.item_score[self.scorer_name]['Default'][idx_list] = scores.cpu().detach().numpy()
            else:
                raise ValueError(f"Invalid scores type {type(scores)} returned by {self.scorer_name}")

        return self.scorer_name, dataset.score_record.item_score[self.scorer_name]
