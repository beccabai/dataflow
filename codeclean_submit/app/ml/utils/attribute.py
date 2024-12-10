import uuid
import os
import numpy as np
import shutil
import re
import json
from typing import List, Callable


def map_attr_file_name(file_name: str, attr_key: str):
    assert attr_key is not None
    bare_file_name = file_name[: file_name.rfind(".")]
    new_name = f"{bare_file_name}-{str(uuid.uuid4()).replace('-', '')[:4]}-attr-{attr_key}.mmap.npy"
    return new_name


class LocalAttributeWriter:
    __tmp_dir = os.path.join("/fs-computility/llm/shared/code-clean/", "local_upload")

    def __init__(
        self,
        path: str,
        post_process_func: Callable,
        dtype: str = None,
        mode: str = "scaler",
        tmp_dir=__tmp_dir,
    ) -> None:
        """
        write attribute in mmap.npy format
        firstly get doc dict sequenlly, then post_process_func to get attribute item
        finally write the attribute item in mmap.npy format to local file system

        Args:
            path (str): is a local file-system path /fs/qiujiantao/task/subpath/part-xxxx-xxxx-attr-key[<flat2d>].mmap.npy
            post_process_func (Callable): post process input doc dict to attribute item
            dtype (str, optional): attribute item dtype. Defaults to None.
            mode (str, optional): attribute mode, scaler or paragraph. Defaults to "scaler".
            tmp_dir (_type_, optional): Defaults to __tmp_dir.
        """

        self.path = path
        self.post_process_func = post_process_func
        self.dtype = dtype

        # now only support scaler mode and flat2d mode
        # for more detail about flat2d mode, please ask WangRui
        assert mode in ["scaler", "flat2d"]
        self.mode = mode

        # check valid path
        self.__check_valid_path(mode, path)

        # use a list as buffer to store attribute item
        self.buffer = []

        # save array to tmp_dir first to make sure the data is complete
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    @staticmethod
    def __check_valid_path(mode: str, path: str) -> bool:
        """
        if in scaler mode, path should as /dir/XXXX-hash-attr-attr_key.mmap.npy
        if in flat2d mode, path should as /dir/XXXX-hash-attr-attr_key<flat2d>.2d.mmap.npy
        finally, the placeholder <flat2d> will be replace to [_off, _len, _flat] then write to
        1. /dir/XXXX-hash-attr-attr_key_off.2d.mmap.npy
        2. /dir/XXXX-hash-attr-attr_key_len.2d.mmap.npy
        3. /dir/XXXX-hash-attr-attr_key_flat.2d.mmap.npy
        """
        filename = os.path.basename(path)

        for post_fix in [".2d.mmap.npy", ".mmap.npy"]:
            if filename.endswith(post_fix):
                path_post_part = filename[: filename.rfind(post_fix)]
            break
        if mode == "flat2d":
            if not path_post_part.endswith("<flat2d>"):
                raise ValueError(f"invalid path {path} for flat2d mode, should end with <flat2d>.2d.mmap.npy")
        else:
            if path_post_part.endswith("<flat2d>"):
                raise ValueError(f"invalid path {path} for scaler mode, should not end with <flat2d>.2d.mmap.npy")

    def get_flat2d_name(self, target_path: str):
        """
        replace <flat2d> to [_off, _len, _flat]
        refer to __check_valid_path
        """
        map_rules = {"len": "_len", "off": "_off", "flat": "_flat"}
        return {key: target_path.replace("<flat2d>", value) for key, value in map_rules.items()}

    def write(self, d: dict):
        d = d.copy()
        attr_result = self.post_process_func(d)
        self.buffer.append(attr_result)
        return None

    def build_array(self, buffer: List):
        """
        build array from buffer
        if in scaler mode, return {"data": np.array}
        if in flat2d mode, return {"len": np.array, "off": np.array, "flat": np.array}
        """
        if self.mode == "scaler":
            if self.dtype is None:
                array = np.array(buffer)
            else:
                array = np.array(buffer, dtype=self.dtype)
            result = {"data": array}
        elif self.mode == "flat2d":
            # for a buffer with [[0.0], [1.0,2.0], [3.0,4.0,5.0]]
            # convert into len, off, flat
            # len = [1, 2, 3]
            # off = [0, 1, 3]
            # flat = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            len_list = [len(line) for line in buffer]
            len_array = np.array(len_list, dtype=np.int32)
            off_array = np.cumsum([0] + len_list, dtype=np.int32)[:-1]
            flat_list = []
            for line in buffer:
                flat_list.extend(line)
            if self.dtype is None:
                flat_array = np.array(flat_list)
            else:
                flat_array = np.array(flat_list, dtype=self.dtype)
            result = {"len": len_array, "off": off_array, "flat": flat_array}

        return result

    def get_tmp_file_name(self, target_path: str):
        # add uuid after the file name
        file_name = os.path.basename(target_path)
        return os.path.join(self.tmp_dir, f"{file_name}.{str(uuid.uuid4())}")

    def flush(self):
        # to make sure the data is complete
        # fisrt build save_task_list
        # then try to save all data to tmp file
        # if all tmp file saved, move all tmp files to target path
        # if any error occurs, remove all target path
        # always remove all tmp files

        all_success = False
        try:
            array_result = self.build_array(self.buffer)
            save_task_list = []
            # make sure save_task_list is not empty
            if self.mode == "scaler":
                # only one file for scaler mode
                tmp_file = self.get_tmp_file_name(self.path)
                save_task_list.append({"data": array_result["data"], "tmp_file": tmp_file, "target_path": self.path})
            else:
                # save len, off, flat to three files
                for key, target_path in self.get_flat2d_name(self.path).items():
                    tmp_file = self.get_tmp_file_name(target_path)
                    save_task_list.append({"data": array_result[key], "tmp_file": tmp_file, "target_path": target_path})
            target_dir = os.path.dirname(self.path)
            os.makedirs(target_dir, exist_ok=True)
            # may fail when save data to tmp file
            for task in save_task_list:
                with open(task["tmp_file"], "wb") as f:
                    np.save(f, task["data"])

            # move all tmp files to target path
            for task in save_task_list:
                src = task["tmp_file"]
                dst = task["target_path"]
                shutil.move(src, dst)

            all_success = True
        finally:
            self.buffer = []
            try:
                if not all_success:
                    for task in save_task_list:
                        if os.path.exists(task["target_path"]):
                            os.remove(task["target_path"])
                for task in save_task_list:
                    if os.path.exists(task["tmp_file"]):
                        os.remove(task["tmp_file"])
            except:
                pass


class JsonlHandler:
    def __init__(self, path: str, config: dict = {}):
        self.path = path
        self.yield_line = config.get("yield_line", False)

    def __iter__(self):
        with open(self.path, "r") as file:
            for line in file:
                if self.yield_line:
                    yield line
                else:
                    dd = json.loads(line)
                    output_dict = {key: dd.get(key, None) for key in ["id", "content"]}
                    yield output_dict


class ScalerAttributeHandler:
    def __init__(self, path: str):
        self.path = path
        self.attr_array = None

    def __enter__(self):
        self.attr_array = np.load(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.attr_array

    def __iter__(self):
        for i in range(len(self.attr_array)):
            yield self.attr_array[i]


default_name_map = {"len": "_len", "off": "_off", "flat": "_flat"}


class Flat2dAttributeHandler:
    def __init__(self, path: str, name_map: dict = default_name_map):
        assert "<flat2d>" in path
        self.path = path
        self.attr_len = None
        self.attr_off = None
        self.attr_flat = None
        self.name_map = name_map if name_map else default_name_map

    def __enter__(self):
        self.attr_len = np.load(self.path.replace("<flat2d>", self.name_map["len"]))
        self.attr_off = np.load(self.path.replace("<flat2d>", self.name_map["off"]))
        self.attr_flat = np.load(self.path.replace("<flat2d>", self.name_map["flat"]))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del [self.attr_len, self.attr_off, self.attr_flat]

    def __iter__(self):
        for i in range(len(self.attr_len)):
            yield self.attr_flat[self.attr_off[i] : self.attr_off[i] + self.attr_len[i]].tolist()


def get_attr_key_from_filename(file_name: str):
    # remove "-attr-" and remained right part
    parttern = re.compile(r".*-attr-", re.I)
    right_part = re.sub(parttern, "", file_name)
    # find the first "." and remove the right part
    return right_part[: right_part.find(".")]


def match_attr_file_by_keys(
    content_path: str, content_root_dir: str, attr_root_dir: str, attr_key_list: List[str], attr_filter: str = "mmap.npy"
):
    attr_file_map = {attr_key: [] for attr_key in attr_key_list}

    content_file_name = os.path.basename(content_path)
    attr_prefix = content_file_name.replace(".jsonl", "")
    content_file_dir = os.path.dirname(content_path)
    full_sub_dir = content_file_dir[len(content_root_dir) :]
    full_sub_dir = full_sub_dir.strip("/")
    attr_full_sub_dir = os.path.join(attr_root_dir, full_sub_dir)
    if not os.path.exists(attr_full_sub_dir):
        return attr_file_map

    attr_file_list = [f for f in os.listdir(attr_full_sub_dir) if (f.startswith(attr_prefix) and attr_filter in f)]
    for attr_file in attr_file_list:
        file_attr_key = get_attr_key_from_filename(attr_file)
        match_keys = [attr_key for attr_key in attr_key_list if attr_key in file_attr_key]
        assert len(match_keys) <= 1
        for key in match_keys:
            attr_file_map[key].append(os.path.join(attr_full_sub_dir, attr_file))
    return attr_file_map


def match_attr_file_auto(content_path: str, content_root_dir: str, attr_root_dir: str, attr_filter: str = "mmap.npy"):
    attr_file_map = {}

    content_file_name = os.path.basename(content_path)
    attr_prefix = content_file_name.replace(".jsonl", "")
    content_file_dir = os.path.dirname(content_path)
    full_sub_dir = content_file_dir[len(content_root_dir) :]
    attr_full_sub_dir = os.path.join(attr_root_dir, full_sub_dir)
    if not os.path.exists(attr_full_sub_dir):
        return attr_file_map

    attr_file_list = [f for f in os.listdir(attr_full_sub_dir) if (f.startswith(attr_prefix) and attr_filter in f)]
    for attr_file in attr_file_list:
        file_attr_key = get_attr_key_from_filename(attr_file)

        if file_attr_key not in attr_file_map:
            attr_file_map[file_attr_key] = []
        attr_file_map[file_attr_key].append(os.path.join(attr_full_sub_dir, attr_file))
    return attr_file_map


class ContentAttributeReader:
    def __init__(
        self, content_path: str, content_root_dir: str, attr_root_dir: str, attr_key_list: List[str] = None, config={}
    ):
        """
        read a jsonl file and the attr file in meta folder and output as an dict
        for a line in jsonl file as {"id": "123", "content": "this is a content"}
        and attr file : XXXX-attr-ad_score.mmap.npy, XXXX-attr-info_score.mmap.npy
        output is {"id": "123", "content": "this is a content", "ad_score": ad_score, "info_score": info_score}

        Args:
            content_path (str): the jsonl file to read
            content_root_dir (str): the content file root dir 
            attr_root_dir (str): the meta file root dir
            attr_key_list (List[str], optional): the attr_key to match, if None, auto match all attr file, if [] match nothing. Defaults to None.
            config (dict, optional): the config for attr file name map. Defaults to {}.
        """
        self.content_path = content_path
        self.content_root_dir = content_root_dir
        if not content_path.startswith(content_root_dir):
            raise ValueError(f"invalid content path {content_path}, must start with {content_root_dir}")
        if not content_path.endswith(".jsonl"):
            raise ValueError(f"invalid content path {content_path}, must end with .jsonl")
        self.attr_root_dir = attr_root_dir
        self.attr_key_list = attr_key_list
        self.calc_line_bytes = config.get("calc_line_bytes", False)
        self.config = config

    def __enter__(self):
        self.jsonl_handlers = JsonlHandler(self.content_path, {"yield_line": self.calc_line_bytes})
        if self.attr_key_list is not None:
            match_result = match_attr_file_by_keys(
                self.content_path, self.content_root_dir, self.attr_root_dir, self.attr_key_list
            )
        else:
            match_result = match_attr_file_auto(self.content_path, self.content_root_dir, self.attr_root_dir)
        self.attr_handlers = {}
        for attr_key, match_files in match_result.items():
            if len(match_files) == 0:
                pass
            elif len(match_files) == 1:
                self.attr_handlers[attr_key] = ScalerAttributeHandler(match_files[0])
                self.attr_handlers[attr_key].__enter__()
            elif len(match_files) == 3:
                # todo build attr handler
                flat_file_name = [f for f in match_files if "flat" in f]
                self.attr_handlers[attr_key] = Flat2dAttributeHandler(
                    flat_file_name[0].replace("_flat", "<flat2d>"), self.config.get("name_map", None)
                )
                self.attr_handlers[attr_key].__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.attr_handlers

    def __iter__(self):
        attr_key_list = list(self.attr_handlers.keys())
        handles = [self.jsonl_handlers] + [self.attr_handlers[attr_key] for attr_key in attr_key_list]
        for attr_list in zip(*handles):
            if self.calc_line_bytes:
                line :str = attr_list[0]
                line_bytes = len(line.encode())
                content_dict = json.loads(line)
                content_dict["line_bytes"] = line_bytes
            else:
                content_dict = attr_list[0]
            result_dict = {attr_key_list[idx]: attr_result for idx, attr_result in enumerate(attr_list[1:])}
            content_dict.update(result_dict)
            yield content_dict


if __name__ == "__main__":
    content_path = "/fs-computility/llm/shared/llm_data/en-cc-90-dedup-0130_v003/raw_data_0130_rearrange/en/CC-MAIN-2017-09/train-65aa8204fc2e-006280.jsonl"
    content_root_dir = "/fs-computility/llm/shared/llm_data/en-cc-90-dedup-0130_v003/raw_data_0130_rearrange/"
    attr_root_dir = "/fs-computility/llm/shared/llm_data/en-cc-90-dedup-0130_v003/raw_data_0130_rearrange_meta/"
    attr_key_list = ["en_coherence_0410", "ad_bert_0203", "en_information_0321", "dedup_para_nums_0530"]
    config = {"name_map": {"len": "_para_nums", "off": "_doc_offset", "flat": "_flat"}}
    idx = 0
    with ContentAttributeReader(content_path, content_root_dir, attr_root_dir, attr_key_list, config) as car:
        for data in car:
            print(data)
            idx += 1
            if idx == 10:
                break
