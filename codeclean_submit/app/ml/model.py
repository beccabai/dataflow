import os
from typing import Generic, List, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from app.common.vector_util import decode_vector, encode_vector

I = TypeVar("I")
B = TypeVar("B")


class Model(Generic[I, B]):
    def __init__(self, device):
        self.device = device

    def preprocess(self, d: dict) -> Tuple[dict, I]:
        """(d: dict) -> (info, input)"""
        raise NotImplementedError()

    def collate_fn(self, lst: List[Tuple[dict, I]]) -> Tuple[List[dict], B]:
        """(lst: list) -> (infos, batch)"""
        raise NotImplementedError()

    def inference(self, batch: B) -> List[dict]:
        """(batch: B) -> results"""
        raise NotImplementedError()


class ConfigModel(Model[I, B]):
    def __init__(self, config: dict, device):
        super().__init__(device)
        self.config = config
        self.model_name = str(config.get("model_name") or "")
        assert self.model_name
        self.model_path = str(config.get("model_path") or "")
        assert self.model_path
        self.output_prefix = str(config.get("output_prefix") or "").rstrip("_")
        self.output_postfix = str(config.get("output_postfix") or "").lstrip("_")

    def get_output_key(self, f: str):
        prefix = self.output_prefix if self.output_prefix else self.model_name
        postfix = f"_{self.output_postfix}" if self.output_postfix else ""
        return f"{prefix}_{f}{postfix}"


class TextModel(ConfigModel[I, B]):
    info_fields = ["track_id", "id", "language"]

    def __init__(self, config: dict, device):
        super().__init__(config, device)
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.max_tokens = int(config.get("max_tokens", 512))

    def load_model(self):
        raise NotImplementedError()

    def load_tokenizer(self):
        raise NotImplementedError()

    def get_doc_info(self, d: dict):
        return {f: d[f] for f in self.info_fields if f in d}


class ImageModel(Model[Tensor, Tensor]):
    def preprocess(self, d: dict) -> Tuple[dict, Tensor]:
        return super().preprocess(d)

    def collate_fn(self, lst: List[Tuple[dict, Tensor]]) -> Tuple[List[dict], Tensor]:
        infos = [item[0] for item in lst]
        tensors = [item[1] for item in lst]
        return infos, torch.stack(tensors)


class BertModel(TextModel[List[int], dict]):
    pad_token_id = 0

    def __init__(self, config: dict, device):
        super().__init__(config, device)
        tail_tokens = int(config.get("tail_tokens") or config.get("remain_tail", 0))
        self.tail_tokens = min(max(0, tail_tokens), self.max_tokens - 1)

    def _config_bert_model(self, model):
        model.eval()

        # TODO: remove legacy config items in future.
        if self.config.get("to_float16") or self.config.get("use_bert_fast") or self.config.get("use_flash_attn"):
            model = model.to(dtype=torch.float16)
        if self.config.get("to_float32"):
            model = model.to(dtype=torch.float32)
        if self.config.get("to_bettertransformer") or self.config.get("use_flash_attn"):
            assert hasattr(model, "to_bettertransformer")
            model = model.to_bettertransformer()

        model = model.to(self.device)
        return model

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_path)

    def preprocess(self, d: dict) -> Tuple[dict, List[int]]:
        from app.common.json_util import Doc

        head_tokens = self.max_tokens - self.tail_tokens
        head_len = head_tokens * 10  # magic number
        tail_len = self.tail_tokens * 10

        if d.get("label_text"):  # temp code
            text = d["label_text"]
        else:
            text = Doc(d).truncated_content(head_len, tail_len)

        tokens = self.tokenizer(text)["input_ids"]
        if len(tokens) > self.max_tokens:
            tokens_tail = tokens[-self.tail_tokens :] if self.tail_tokens else []
            tokens = tokens[:head_tokens] + tokens_tail

        return (self.get_doc_info(d), tokens)

    def collate_fn(self, lst: List[Tuple[dict, List[int]]]) -> Tuple[List[dict], dict]:
        infos = [item[0] for item in lst]
        tokens_list = [item[1] for item in lst]
        max_len = max(len(t) for t in tokens_list)

        input_ids, attention_mask = [], []
        for tokens in tokens_list:
            tok_len = len(tokens)
            pad_len = max_len - tok_len
            input_ids.append(tokens + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * tok_len + [0] * pad_len)

        inputs = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
        return infos, inputs


class BertTwoClassModel(BertModel):
    def __init__(self, config: dict, device):
        super().__init__(config, device)
        self.prob_key = self.get_output_key("prob")
        self.cls_index = int(config.get("cls_index", 1))  # use `1` as default value is not good.
        self.use_sigmoid = bool(config.get("use_sigmoid", False))
        self.use_logits = bool(config.get("use_logits", False))
        self.use_clip = bool(config.get("use_clip", False))
        self.clip_min = float(config.get("clip_min", 0))
        self.clip_max = float(config.get("clip_max", 1))

    def load_model(self):
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        return self._config_bert_model(model)

    def inference(self, batch: dict) -> List[dict]:
        batch = {k: t.to(self.device) for k, t in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs.logits

            if self.use_logits:
                prob = logits
            elif self.use_sigmoid:
                prob = torch.sigmoid(logits)
            else:
                prob = torch.softmax(logits, dim=-1)

            if self.use_clip:
                prob = prob.clip(self.clip_min, self.clip_max)

            cls_index = min(self.cls_index, prob.shape[1] - 1)
            cls_prob_arr = prob[:, cls_index].detach().cpu().numpy()

        return [{self.prob_key: round(float(p), 6)} for p in cls_prob_arr]

        # scores = logits.squeeze(-1).float().detach().cpu().numpy().tolist()
        # int_scores = [int(round(max(0, min(score, 5)))) for score in scores]
        # for score in int_scores:
        #     output = {self.get_output_key("prob"): score}

class BertClassRegressionModel(BertModel):
    def __init__(self, config: dict, device):
        super().__init__(config, device)
        self.prob_key = self.get_output_key("prob")
        self.cls_index = int(config.get("cls_index", 1))  # use `1` as default value is not good.
        self.use_sigmoid = bool(config.get("use_sigmoid", False))
        self.use_logits = bool(config.get("use_logits", False))
        self.use_clip = bool(config.get("use_clip", False))
        self.clip_min = float(config.get("clip_min", 0))
        self.clip_max = float(config.get("clip_max", 1))

    def load_model(self):
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        return self._config_bert_model(model)

    def inference(self, batch: dict) -> List[dict]:
        batch = {k: t.to(self.device) for k, t in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs.logits

            if self.use_logits:
                prob = logits

            if self.use_clip:
                prob = prob.clip(self.clip_min, self.clip_max)

            # cls_index = min(self.cls_index, prob.shape[1] - 1)
            # cls_prob_arr = prob[:, cls_index].detach().cpu().numpy()
            cls_prob_arr = prob.detach().cpu().numpy()

        return [{self.prob_key: round(float(p), 6)} for p in cls_prob_arr]

        # scores = logits.squeeze(-1).float().detach().cpu().numpy().tolist()
        # int_scores = [int(round(max(0, min(score, 5)))) for score in scores]
        # for score in int_scores:
        #     output = {self.get_output_key("prob"): score}


class BertEncodeModel(BertModel):
    def __init__(self, config: dict, device):
        super().__init__(config, device)
        self.features_key = self.get_output_key("features")
        self.encode_features = bool(config.get("encode_features", True))

    def load_model(self):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.model_path)
        return self._config_bert_model(model)

    def inference(self, batch: dict) -> List[dict]:
        batch = {k: t.to(self.device) for k, t in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            last_hidden_state = outputs[0]
            bos_features = last_hidden_state[:, 0]
            bos_features = F.normalize(bos_features, dim=-1)
            bos_features_arr = bos_features.detach().cpu().numpy()

        if self.encode_features:
            bos_features_arr = [encode_vector(f) for f in bos_features_arr]

        return [{self.features_key: f} for f in bos_features_arr]


def _read_file(path: str):
    from app.common.asset_util import get_asset
    from app.common.s3 import is_s3_path, read_s3_object_bytes

    if is_s3_path(path):
        return read_s3_object_bytes(path)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    with open(get_asset(path), "rb") as f:
        return f.read()


def _l2_distance(x: Tensor, y: Tensor):
    """use sqrt(x**2 - 2xy + y**2)"""
    x_square = (x**2).sum(dim=-1, keepdim=True)
    y_square = (y**2).sum(dim=-1, keepdim=True)
    dist = x_square - 2 * (x @ y.T) + y_square.T
    return dist.clamp(min=0.0).sqrt()


class BertEncodeCidModel(BertEncodeModel):
    def __init__(self, config: dict, device):
        super().__init__(config, device)
        cid_path = config.get("cid_path") or config.get("c_id_embedding") or ""
        assert cid_path
        self.cid_key = config.get("cid_key", "c_embs")
        self.cid_similarity = config.get("cid_similarity", "l2_norm")
        assert self.cid_similarity in ("l2_norm", "dot_product", "cosine")
        value_name = "distance" if self.cid_similarity == "l2_norm" else "similarity"
        self.index_key = self.get_output_key("cid")
        self.value_key = self.get_output_key(value_name)
        self.cid_tensor = self._load_cid_tensor(cid_path)

    def _load_cid_tensor(self, cid_path: str):
        from app.common.json_util import json_loads

        cid_content = _read_file(cid_path).decode("utf-8")
        cid_lines = (l for l in cid_content.split("\n") if l)
        cid_vectors = [decode_vector(json_loads(l)[self.cid_key]) for l in cid_lines]
        cid_tensor = torch.tensor(np.array(cid_vectors))
        if self.cid_similarity == "cosine":
            cid_tensor = F.normalize(cid_tensor, dim=-1)
        return cid_tensor.to(self.device)

    def _calc_with_cid(self, features: Tensor):
        features = features.to(self.cid_tensor.dtype)
        if self.cid_similarity == "l2_norm":
            distances = _l2_distance(features, self.cid_tensor)
            cid_indices = torch.argmin(distances, dim=-1)
            cid_values = distances[torch.arange(distances.size(0)), cid_indices]
        else:  # cid_similarity in (dot_product, cosine)
            similarities = features @ self.cid_tensor.T
            cid_indices = torch.argmax(similarities, dim=-1)
            cid_values = similarities[torch.arange(similarities.size(0)), cid_indices]
        return cid_indices, cid_values

    def inference(self, batch: dict) -> List[dict]:
        batch = {k: t.to(self.device) for k, t in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            last_hidden_state = outputs[0]
            bos_features = last_hidden_state[:, 0]
            bos_features = F.normalize(bos_features, dim=-1)
            cid_indices, cid_values = self._calc_with_cid(bos_features)
            bos_features_arr = bos_features.detach().cpu().numpy()
            cid_indices_arr = cid_indices.detach().cpu().numpy()
            cid_values_arr = cid_values.detach().cpu().numpy()

        if self.encode_features:
            bos_features_arr = [encode_vector(f) for f in bos_features_arr]

        return [
            {
                self.features_key: features,
                self.index_key: int(cid_indices_arr[idx]),
                self.value_key: round(float(cid_values_arr[idx]), 6),
            }
            for idx, features in enumerate(bos_features_arr)
        ]
