import math
import os
import zipfile
from dataclasses import dataclass
from typing import Callable, List, Tuple

import open_clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.common.s3 import download_s3_file_with_retry
from app.common.vector_util import encode_vector
from app.ml.model import ImageModel

MODEL_DIR = "mm_models"


@dataclass
class ModelInfo:
    name: str
    s3_file: str
    local_file: str
    local_dir: str = ""

    def download(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if not os.path.exists(self.local_file):
            print(f"Downloading {self.name} ...")
            download_s3_file_with_retry(self.s3_file, self.local_file)
        if self.local_dir and not os.path.exists(self.local_dir):
            print(f"Extracting {self.name} ...")
            os.makedirs(self.local_dir, exist_ok=True)
            with zipfile.ZipFile(self.local_file, "r") as z:
                z.extractall(MODEL_DIR)


CLIP_MODEL = ModelInfo(
    name="ViT-L-14",
    s3_file="s3://llm-pipeline/ASSETS/ViT-L-14.pt",
    local_file=os.path.join(MODEL_DIR, "ViT-L-14.pt"),
)
AESTHETIC_MODEL = ModelInfo(
    name="improved-aesthetic-predictor",
    s3_file="s3://llm-pipeline/ASSETS/sac+logos+ava1-l14-linearMSE.pth",
    local_file=os.path.join(MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth"),
)
NSFW_MODEL = ModelInfo(
    name="CLIP-based-NSFW-Detector",
    s3_file="s3://llm-pipeline/ASSETS/clip_autokeras_binary_nsfw.zip",
    local_file=os.path.join(MODEL_DIR, "clip_autokeras_binary_nsfw.zip"),
    local_dir=os.path.join(MODEL_DIR, "clip_autokeras_binary_nsfw"),
)
WATERMARK_MODEL = ModelInfo(
    name="efficientnet_b3",
    s3_file="s3://llm-pipeline/ASSETS/efficientnet_b3a.safetensors",
    local_file=os.path.join(MODEL_DIR, "efficientnet_b3a.safetensors"),
)
WATERMARK_CLASSIFIER = ModelInfo(
    name="watermark_model_v1.pt",
    s3_file="s3://llm-pipeline/ASSETS/watermark_model_v1.pt",
    local_file=os.path.join(MODEL_DIR, "watermark_model_v1.pt"),
)
QRCODE_CLASSIFIER = ModelInfo(
    name="svm_qr_240830.pkl",
    s3_file="s3://llm-pipeline/ASSETS/svm_qr_240830.pkl",
    local_file=os.path.join(MODEL_DIR, "svm_qr_240830.pkl"),
)


class AestheticPredictor(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def load_aesthetic_model(device):
    model = AestheticPredictor(768)
    state_dict = torch.load(AESTHETIC_MODEL.local_file)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def disable_tensorflow_gpus():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"


def load_nsfw_model():
    import autokeras as ak
    from tensorflow.keras.models import load_model  # type: ignore

    model = load_model(
        NSFW_MODEL.local_dir,
        custom_objects=ak.CUSTOM_OBJECTS,
        compile=False,
    )
    return model


def load_watermark_model(device):
    model = timm.create_model(
        WATERMARK_MODEL.name,
        pretrained=True,
        pretrained_cfg={"file": WATERMARK_MODEL.local_file},
    )

    model.classifier = nn.Sequential(
        # 1536 is the original in_features
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    state_dict = torch.load(WATERMARK_CLASSIFIER.local_file)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_qrcode_classifier():
    from pickle import load

    with open(QRCODE_CLASSIFIER.local_file, "rb") as f:
        classifier = load(f)
    return classifier


def clip_classifier(text1, text2, model, device, tokenizer):
    "return prob of text1"
    texts = tokenizer([text1, text2])
    texts = texts.to(device)

    with torch.no_grad():
        text_features = model.encode_text(texts)
        text_features = F.normalize(text_features, dim=-1)

    def predict(image_features: torch.Tensor):
        pred = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return pred[:, 0].tolist()

    return predict


def get_transform():
    from open_clip.transform import PreprocessCfg, image_transform_v2

    transform = image_transform_v2(
        cfg=PreprocessCfg(size=(224, 224)),
        is_train=False,
    )
    return transform


def similarities(image_features: torch.Tensor, text_features: torch.Tensor):
    pred = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return pred.tolist()


def round_prob(val: float):
    if math.isnan(val):
        return 0.0
    if math.isinf(val):
        return 1.0 if val > 0 else 0.0
    return round(val, 6)


class ImgClassifier(ImageModel):
    def __init__(self, device=None, encode_features=True):
        super().__init__(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encode_features = encode_features

        # Disable GPU, cuz gpu-memory issue.
        disable_tensorflow_gpus()

        CLIP_MODEL.download()
        AESTHETIC_MODEL.download()
        NSFW_MODEL.download()
        WATERMARK_MODEL.download()
        WATERMARK_CLASSIFIER.download()
        QRCODE_CLASSIFIER.download()

        model, _, preprocess_val = open_clip.create_model_and_transforms(
            CLIP_MODEL.name,
            pretrained=CLIP_MODEL.local_file,
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL.name)

        assert isinstance(model, open_clip.CLIP)
        assert isinstance(preprocess_val, Callable)

        self.model, self.preprocess_val = model, preprocess_val
        self.model.to(self.device)
        self.model.eval()

        self.aesthetic_model = load_aesthetic_model(self.device)
        self.nsfw_model = load_nsfw_model()
        self.watermark_model = load_watermark_model(self.device)
        self.qrcode_classifier = load_qrcode_classifier()

    def preprocess(self, d: dict) -> Tuple[dict, torch.Tensor]:
        img_tensor = self.preprocess_val(d.pop("img"))
        assert isinstance(img_tensor, torch.Tensor)
        return d, img_tensor

    def inference(self, batch: torch.Tensor) -> List[dict]:
        batch = batch.to(self.device)

        with torch.no_grad(), torch.autocast(self.device):  # type: ignore
            image_features = self.model.encode_image(batch, normalize=True)
            image_features_arr = image_features.detach().cpu().numpy()

            aesthetic_score = self.aesthetic_model(image_features).view(-1).tolist()
            nsfw_prob = self.nsfw_model.predict(image_features_arr, verbose=0).reshape(-1).tolist()
            watermark_prob = F.softmax(self.watermark_model(batch), dim=1)[:, 0].tolist()
            has_qrcode_cls = self.qrcode_classifier.predict(image_features_arr)

        ret = []
        for i in range(len(batch)):
            item_features = image_features_arr[i]
            if self.encode_features:
                item_features = encode_vector(item_features)
            ret.append(
                {
                    "image_features": item_features,
                    "aesthetic_score": round_prob(aesthetic_score[i]),
                    "nsfw_prob": round_prob(nsfw_prob[i]),
                    "watermark_prob": round_prob(watermark_prob[i]),
                    "has_qrcode_cls": bool(has_qrcode_cls[i]),
                }
            )
        return ret

    def inference_imgs(self, img_list: list):
        if not img_list:
            return []
        tensor_list = []
        for img in img_list:
            img_tensor = self.preprocess_val(img)
            assert isinstance(img_tensor, torch.Tensor)
            tensor_list.append(img_tensor)
        batch = torch.stack(tensor_list)
        return self.inference(batch)

    def encode_text(self, text: str):
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: list):
        tokens = self.tokenizer(texts)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens, normalize=True)
            text_features_arr = text_features.detach().cpu().numpy()

        return text_features_arr


"""
import clip
from functools import partial

# clip.available_models()

proxy_on()
model2, preprocess2 = clip.load("ViT-L/14", device=device)
tokenizer2 = partial(clip.tokenize, truncate=True)
# tokenizer = TokenizeTransformWrapper(partial(openai_tokenize, context_length=77))
proxy_off()

model2.to(device)
model2.eval()
"""
