import os
import shutil
import uuid
from typing import Union
from app.common.json_util import json_encode

class LocalDocWriter:
    
    __tmp_dir = os.path.join("/fs-computility/llm/shared/code-clean/", "local_upload")

    def __init__(
        self,
        path: str,
        tmp_dir=__tmp_dir,
    ) -> None:
        self.path = path

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

    def write(self, d: dict):
        d = d.copy()

        doc_bytes = json_encode(d)

        self.tmp_fh.write(doc_bytes)
        self.offset += len(doc_bytes)

        return len(doc_bytes)

    def flush(self):
        try:
            self.tmp_fh.close()
            dir = os.path.dirname(self.path)
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
            shutil.move(self.tmp_file, self.path)
        finally:
            try:
                os.remove(self.tmp_file)
            except:
                pass


def list_local_fs_files_recursively(root_dir: str, include_postfix: Union[str, list, None] = None):
    """
    List all files in the root_dir recursively. only include files with postfix in include_postfix
    
    Args:
        root_dir (str): the root directory to list recursively
        include_postfix (Union[str, list, None]): the postfix of the files to include. Defaults to None.

    Returns:
        list: a list of file paths
    """
    if isinstance(include_postfix, str):
        include_postfix_list = [include_postfix]
    elif isinstance(include_postfix, list):
        include_postfix_list = include_postfix
    elif include_postfix is None:
        include_postfix_list = []
    else:
        raise ValueError(f"include_postfix should be a list of str or None, got {include_postfix}")
    
    matched_files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            include_flag = False
            if include_postfix_list is None:
                include_flag = True
            else:
                if any([filename.endswith(postfix) for postfix in include_postfix_list]):
                    include_flag = True
            if include_flag:
                matched_files.append(os.path.join(root, filename))
    return matched_files