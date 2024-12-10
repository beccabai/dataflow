import ast
import json
import re
from typing import List, Union

try:
    import orjson
except:
    orjson = None


_surrogates_re = r"[\ud800-\udfff]"


def json_loads(s: Union[str, bytes], **kwargs) -> dict:
    if not kwargs and orjson:
        try:
            return orjson.loads(s)
        except:
            pass
    try:
        return json.loads(s, **kwargs)
    except Exception as e:
        if "enclosed in double quotes" not in str(e):
            raise e
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        else:
            s = str(s)
        return ast.literal_eval(s)


def json_dumps(d: dict, **kwargs) -> str:
    if not kwargs and orjson:
        try:
            return orjson.dumps(d).decode("utf-8")
        except:
            pass
    return json.dumps(d, ensure_ascii=False, **kwargs)


def str_encode(s: str) -> bytes:
    # try remote special characters
    s = re.sub(_surrogates_re, "\ufffd", s)

    try:
        return s.encode("utf-8")
    except UnicodeEncodeError as e:
        debug_start = max(0, e.start - 1000)
        debug_end = min(len(s), e.end + 1000)
        print(f"{debug_start=}, {debug_end=}, debug_s={s[debug_start:debug_end]}")
        raise


def json_encode(d: dict, end="\n", **kwargs) -> bytes:
    return str_encode(json_dumps(d, **kwargs) + end)


def json_print(obj):
    if isinstance(obj, list) and len(obj):
        return json_print(obj[0])
    if isinstance(obj, bytes):
        return json_print(obj.decode("utf-8"))
    if isinstance(obj, str):
        return json_print(json_loads(obj))
    if isinstance(obj, dict):
        return print(json_dumps(obj, indent=2))

    try:
        from pyspark.sql import Row
    except:
        Row = None

    if Row and isinstance(obj, Row) and "value" in obj:
        return json_print(obj.value)

    print(obj)


def safe_get(jso: dict, key: str, default_val=""):
    v = jso.get(key, default_val)
    if v is None:
        v = default_val
    return v


def drop_dict(d: dict, reason: str = ""):
    d["dropped"] = True
    if len(reason) != 0:
        d["dropped_by"] = reason
        reason = reason.replace(" ", "_")
        d[f"is_dropped_by_{reason}"] = True


class CItem(dict):
    pass


class Doc(dict):
    def __init__(self, *args, **kwargs):
        # fmt: off
        if (len(args) > 0
            and isinstance(args[0], str)
            and len(args[0]) > 0
            and args[0][0] == "{"
        ):
            # fmt: on
            args = (json_loads(args[0]), *args[1:])

        super().__init__(*args, **kwargs)

    @property
    def content_list(self) -> List[CItem]:
        if self.get("content_list") is None:
            self["content_list"] = []
        if not isinstance(self["content_list"], list):
            self["content_list"] = [self["content_list"]]
        content_list = self["content_list"]
        for i in range(len(content_list)):
            item = content_list[i]
            if isinstance(item, CItem):
                continue
            if not isinstance(item, dict):
                item = {"$val": item}
            content_list[i] = CItem(item)
        return content_list

    def process_content_item(self, item: dict):
        # from app.common_clean.core.definitions import DROP_ITEM_INFO_LIST_KEY
        DROP_ITEM_INFO_LIST_KEY = "drop_it_list"
        from app.format.html import md_to_text

        if item.get(DROP_ITEM_INFO_LIST_KEY):
            return
        if item.get("type") == "hr":
            return
        if "md" in item:
            text = item["md"]
            text_is_md = True
        elif "text" in item:
            text = item["text"]
            text_is_md = bool(item.get("text_format") == "md")
        else:
            return
        if text_is_md:
            text = md_to_text(text)
        return text

    @property
    def content(self) -> str:

        texts = []
        content_list = self.content_list
        for item in content_list:
            text = self.process_content_item(item)
            if text:
                texts.append(text)

        if content_list:
            return "\n\n".join(texts)

        # fallback to old standard
        if self.get("content"):
            return self["content"]

        # final guess
        return self.get("text") or self.get("string") or ""

    def truncated_content(self, head_len: int, tail_len: int) -> str:

        text = ""
        if self.content_list:
            content_list = self.content_list
            idx = -1
            heads = []
            tails = []

            if head_len > 0:
                head_count = 0
                for idx, item in enumerate(content_list):
                    text = self.process_content_item(item)
                    if not text:
                        continue
                    heads.append(text)
                    head_count += len(text)

            if tail_len > 0:
                tail_count = 0
                for i in range(len(content_list) - 1, idx, -1):
                    text = self.process_content_item(content_list[i])
                    if not text:
                        continue
                    tails.append(text)
                    tail_count += len(text)
                    if tail_count >= tail_len:
                        break

            tails.reverse()
            text = "\n\n".join(heads + tails)

        else:
            fields = ["content", "text", "string"]
            for field in fields:
                text = self.get(field, "")
                if text:
                    break

        if head_len + tail_len >= len(text):
            return text
        return text[:head_len] + text[-tail_len:]

    def content_tokens(self, keep_nl=False, skip_punc=False, skip_stop_words=False, to_lower=False, n_gram_size=1) -> List[str]:
        content = self.content
        language = self.get("language") or ""

        if "code" in language:
            # TODO
            pass

        from app.common.clean_tokenizer import content_tokens, ngrams

        tokens = content_tokens(content, keep_nl=keep_nl, skip_punc=skip_punc)
        if to_lower:
            tokens = [t.lower() for t in tokens]
        if n_gram_size > 1:
            tokens = [" ".join(v) for v in ngrams(tokens, n_gram_size)]
        return tokens

    def dump(self) -> str:
        return json_dumps(self)
