import codecs
import io
from base64 import b64encode
from typing import Union

from app.common.json_util import json_dumps

_kept_warc_headers = ["WARC-IP-Address", "WARC-Identified-Payload-Type"]
_html_types = ["text/html", "application/xhtml+xml"]


def _is_valid_charset(charset: str):
    try:
        codecs.lookup(charset)
        return True
    except LookupError:
        return False


def _try_decode(body_bytes: bytes, http_charset: Union[str, None]):
    import cchardet

    tried_charsets = set()
    # 1. try decode with `http_charset`.
    if http_charset and _is_valid_charset(http_charset):
        try:
            http_charset = http_charset.lower()
            tried_charsets.add(http_charset)
            return body_bytes.decode(http_charset), http_charset
        except:
            pass
    # 2. try decode with utf-8.
    try:
        tried_charsets.add("utf-8")
        return body_bytes.decode("utf-8"), "utf-8"
    except:
        pass
    # 3. try detect charset and decode.
    charset = cchardet.detect(body_bytes).get("encoding")
    if charset:
        charset = charset.lower()
        if charset in ["gbk", "gb2312"]:
            charset = "gb18030"
        if charset not in tried_charsets:
            try:
                return body_bytes.decode(charset), charset
            except:
                pass
    # 4. gave up.
    return "", ""


def _is_html_payload(warc_headers: dict, http_headers: dict):
    if warc_headers.get("WARC-Identified-Payload-Type") in _html_types:
        return True
    for key, val in http_headers.items():
        if key.lower() == "content-type":
            content_type = val.lower()
            for html_type in _html_types:
                if html_type in content_type:
                    return True
    return False


def read_warc_records(path: str, stream: io.IOBase):
    from fastwarc.warc import ArchiveIterator, WarcRecordType

    pending_record = None

    for record in ArchiveIterator(stream):
        if pending_record is not None:
            length = int(record.stream_pos) - pending_record[1]
            yield (*pending_record, length)
            pending_record = None

        if record.record_type != WarcRecordType.response:
            continue

        warc_headers = {k: record.headers[k] for k in _kept_warc_headers if k in record.headers}

        if record.http_headers is not None:
            http_headers = {str(k): v for k, v in record.http_headers}
            status_code = record.http_headers.status_code
        else:
            http_headers = {}
            status_code = None

        if not _is_html_payload(warc_headers, http_headers):
            continue
        if status_code is None or status_code >= 400:
            continue

        record_id = str(record.record_id).split(":")[-1][:36]
        record_url = record.headers.get("WARC-Target-URI", "")
        record_date = int(record.record_date.timestamp())
        record_offset = int(record.stream_pos)

        d = {
            "track_id": record_id,
            "url": record_url,
            "status": status_code,
            "response_header": http_headers,
            "date": record_date,
        }

        try:
            body_bytes = record.reader.read()
        except:
            d["content_length"] = -1
            body_bytes = None

        if body_bytes is not None:
            try:
                # avoid ValueError: embedded null character
                http_charset = record.http_charset
            except:
                pass
            html, charset = _try_decode(body_bytes, http_charset)
            if charset:
                d["content_length"] = len(body_bytes)
                d["content_charset"] = charset
                d["html"] = html
            else:
                d["content_length"] = len(body_bytes)
                d["body_bytes"] = b64encode(body_bytes).decode("utf-8")

        d["remark"] = {"warc_headers": warc_headers}
        pending_record = (json_dumps(d), record_offset)

    if pending_record is not None:
        length = stream.tell() - pending_record[1]
        yield (*pending_record, length)
