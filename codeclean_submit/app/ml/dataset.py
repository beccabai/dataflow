import base64
import io
import warnings

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from app.common.json_util import json_loads
from app.common.s3 import read_s3_object_bytes, read_s3_object_io, read_s3_rows


def open_image(d: dict, img_data: bytes):
    try:
        img = Image.open(io.BytesIO(img_data))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = img.convert("RGB")
        d["img"] = img
    except Exception as e:
        print(f"Open image error, {d=} {e=}")
        # use blank image as fallback.
        d["img"] = Image.new("RGB", (150, 150), "white")
        d["error"] = "open_image_error"
    return d


class PathDataset(Dataset):
    def __init__(self, path: str, preprocess):
        self.path = path
        self.preprocess = preprocess


class S3JsonlDataset(PathDataset):
    def __init__(self, path: str, preprocess, jsonl_s3_client=None):
        super().__init__(path, preprocess)
        self.rows = list(read_s3_rows(path, client=jsonl_s3_client))

    def __getitem__(self, index):
        row = self.rows[index]
        d = json_loads(row.value)
        return self.preprocess(d)

    def __len__(self):
        return len(self.rows)


class JsonlDataset(PathDataset):
    def __init__(self, path: str, preprocess):
        super().__init__(path, preprocess)
        with open(path, "r") as f:
            self.rows = f.readlines()

    def __getitem__(self, index):
        row = self.rows[index]
        d = json_loads(row)
        return self.preprocess(d)

    def __len__(self):
        return len(self.rows)


class S3ImageJsonlDataset(S3JsonlDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        d = json_loads(row.value)
        if d.get("img_data"):
            img_data = base64.b64decode(d["img_data"])
        elif d.get("path"):
            img_data = read_s3_object_bytes(d["path"])
        else:
            raise Exception("invalid image json")
        d = open_image(d, img_data)
        return self.preprocess(d)


class S3ImageH5Dataset(PathDataset):
    def __init__(self, path: str, preprocess):
        super().__init__(path, preprocess)
        self.rows = self._read_file(path)

    def _read_file(self, path: str):
        import h5py

        ret = []
        with read_s3_object_io(path) as stream:
            with h5py.File(stream, "r") as f:
                for img_id, data in f.items():
                    img_data = np.array(data).tobytes()
                    ret.append((img_id, img_data))
        return ret

    def __getitem__(self, index):
        img_id, img_data = self.rows[index]
        d = {"id": img_id}
        d = open_image(d, img_data)
        return self.preprocess(d)

    def __len__(self):
        return len(self.rows)
