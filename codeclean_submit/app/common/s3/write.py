import bz2
import gzip
import io
import os
import uuid

import numpy as np

from app.common.const import FIELD_ID
from app.common.json_util import json_encode
from app.common.s3.retry import s3_upload_with_retry

_compressions = {
    "gz": "gz",
    "gzip": "gz",
    "bz2": "bz2",
    "bzip": "bz2",
    "bzip2": "bz2",
    "raw": "raw",
    "none": "raw",
}


class S3DocWriter:
    __tmp_dir = os.path.join(".", "s3_upload")

    def __init__(
        self,
        path: str,
        client=None,
        tmp_dir=__tmp_dir,
        skip_loc=False,
        compression="",
        filename="",
        flush=False,
    ) -> None:
        if not path.startswith("s3://"):
            raise Exception(f"invalid s3 path [{path}].")

        compression = _compressions.get(compression)
        if compression and not path.endswith(f".{compression}"):
            raise Exception(f"path must endswith [.{compression}]")
        if not compression and path.endswith(".gz"):
            compression = "gz"
        if not compression and path.endswith(".bz2"):
            compression = "bz2"

        self.path = path
        self.client = client
        self.skip_loc = skip_loc
        self.compression = compression
        self._flush = flush

        os.makedirs(tmp_dir, exist_ok=True)

        ext = self.__get_ext(path)
        if filename:
            self.tmp_file = os.path.join(tmp_dir, filename)
        else:
            self.tmp_file = os.path.join(tmp_dir, f"{str(uuid.uuid4())}.{ext}")
        self.tmp_fh = open(self.tmp_file, "ab")
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.flush()

    @staticmethod
    def __get_ext(path: str):
        filename = os.path.basename(path)
        parts = filename.split(".")
        if len(parts) > 1 and parts[0]:
            return parts[-1]
        return "txt"

    def write(self, d: dict):
        d = d.copy()

        if not self.skip_loc and "doc_loc" in d:
            track_loc = d.get("track_loc") or []
            track_loc.append(d["doc_loc"])
            d["track_loc"] = track_loc

        if self.compression == "gz":
            if not self.skip_loc and FIELD_ID in d:
                d["doc_loc"] = f"{self.path}?bytes={self.offset},0"
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as f:
                f.write(json_encode(d))
            doc_bytes = buf.getvalue()

        elif self.compression == "bz2":
            if not self.skip_loc and FIELD_ID in d:
                d["doc_loc"] = f"{self.path}?bytes={self.offset},0"
            buf = io.BytesIO()
            with bz2.BZ2File(buf, mode="wb") as f:
                f.write(json_encode(d))
            doc_bytes = buf.getvalue()

        else:
            doc_bytes = json_encode(d)

            # add doc_loc if doc has id
            if not self.skip_loc and FIELD_ID in d:
                doc_len, last_len = len(doc_bytes), 0
                while doc_len != last_len:
                    d["doc_loc"] = f"{self.path}?bytes={self.offset},{doc_len}"
                    doc_bytes = json_encode(d)
                    doc_len, last_len = len(doc_bytes), doc_len

        self.tmp_fh.write(doc_bytes)
        self.offset += len(doc_bytes)
        if self._flush:
            self.tmp_fh.flush()
        return len(doc_bytes)

    def flush(self):
        try:
            self.tmp_fh.close()
            s3_upload_with_retry(self.path, self.tmp_file, self.client)
        finally:
            try:
                os.remove(self.tmp_file)
            except:
                pass


class S3LineWriter:
    __tmp_dir = os.path.join(".", "s3_upload")

    def __init__(
        self,
        path: str,
        client=None,
        tmp_dir=__tmp_dir,
        compression="",
    ) -> None:
        if not path.startswith("s3://"):
            raise Exception(f"invalid s3 path [{path}].")

        compression = _compressions.get(compression)
        if compression and not path.endswith(f".{compression}"):
            raise Exception(f"path must endswith [.{compression}]")
        if not compression and path.endswith(".gz"):
            compression = "gz"
        if not compression and path.endswith(".bz2"):
            compression = "bz2"

        self.path = path
        self.client = client
        self.compression = compression

        os.makedirs(tmp_dir, exist_ok=True)

        ext = self.__get_ext(path)
        self.tmp_file = os.path.join(tmp_dir, f"{str(uuid.uuid4())}.{ext}")
        self.tmp_fh = open(self.tmp_file, "ab")
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.flush()

    @staticmethod
    def __get_ext(path: str):
        filename = os.path.basename(path)
        parts = filename.split(".")
        if len(parts) > 1 and parts[0]:
            return parts[-1]
        return "txt"

    def write(self, line: str):
        if line == "" or line[-1] != "\n":
            line += "\n"

        line_bytes = line.encode("utf-8")

        if self.compression == "gz":
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as f:
                f.write(line_bytes)
            line_bytes = buf.getvalue()

        elif self.compression == "bz2":
            buf = io.BytesIO()
            with bz2.BZ2File(buf, mode="wb") as f:
                f.write(line_bytes)
            line_bytes = buf.getvalue()

        self.tmp_fh.write(line_bytes)
        self.offset += len(line_bytes)

        return len(line_bytes)

    def flush(self):
        try:
            self.tmp_fh.close()
            s3_upload_with_retry(self.path, self.tmp_file, self.client)
        finally:
            try:
                os.remove(self.tmp_file)
            except:
                pass


class S3BytesWriter:
    __tmp_dir = os.path.join(".", "s3_upload")

    def __init__(
        self,
        path: str,
        client=None,
        tmp_dir=__tmp_dir,
        compression="",
    ) -> None:
        if not path.startswith("s3://"):
            raise Exception(f"invalid s3 path [{path}].")

        compression = _compressions.get(compression)
        if compression and compression != "raw":
            if not path.endswith(f".{compression}"):
                raise Exception(f"path must endswith [.{compression}]")
        if not compression and path.endswith(".gz"):
            compression = "gz"
        if not compression and path.endswith(".bz2"):
            compression = "bz2"

        self.path = path
        self.client = client
        self.compression = compression

        os.makedirs(tmp_dir, exist_ok=True)

        ext = self.__get_ext(path)
        self.tmp_file = os.path.join(tmp_dir, f"{str(uuid.uuid4())}.{ext}")
        self.tmp_fh = open(self.tmp_file, "ab")
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.flush()

    @staticmethod
    def __get_ext(path: str):
        filename = os.path.basename(path)
        parts = filename.split(".")
        if len(parts) > 1 and parts[0]:
            return parts[-1]
        return "bin"

    def write(self, bytes: bytes):
        if self.compression == "gz":
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as f:
                f.write(bytes)
            bytes = buf.getvalue()

        elif self.compression == "bz2":
            buf = io.BytesIO()
            with bz2.BZ2File(buf, mode="wb") as f:
                f.write(bytes)
            bytes = buf.getvalue()

        self.tmp_fh.write(bytes)
        self.offset += len(bytes)

        return len(bytes)

    def flush(self):
        try:
            self.tmp_fh.close()
            s3_upload_with_retry(self.path, self.tmp_file, self.client)
        finally:
            try:
                os.remove(self.tmp_file)
            except:
                pass


class S3Hdf5Writer:
    __tmp_dir = os.path.join(".", "s3_upload")

    def __init__(
        self,
        path: str,
        client=None,
        tmp_dir=__tmp_dir,
    ) -> None:
        import h5py

        if not path.startswith("s3://"):
            raise Exception(f"invalid s3 path [{path}].")
        if not (path.endswith(".h5") or path.endswith(".hdf5")):
            raise Exception("extension must be .h5 or .hdf5")

        self.path = path
        self.client = client
        os.makedirs(tmp_dir, exist_ok=True)

        ext = self.__get_ext(path)
        self.tmp_file = os.path.join(tmp_dir, f"{str(uuid.uuid4())}.{ext}")
        self.tmp_fh = h5py.File(self.tmp_file, "w")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.flush()

    @staticmethod
    def __get_ext(path: str):
        filename = os.path.basename(path)
        parts = filename.split(".")
        if len(parts) > 1 and parts[0]:
            return parts[-1]
        return "h5"

    def write(self, name, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        if isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.uint8)
        self.tmp_fh.create_dataset(name, data=data)
        return 0

    def flush(self):
        try:
            self.tmp_fh.close()
            s3_upload_with_retry(self.path, self.tmp_file, self.client)
        finally:
            try:
                os.remove(self.tmp_file)
            except:
                pass
