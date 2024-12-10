from app.common.s3.client import get_s3_client, split_s3_path


def get_s3_presigned_url(path: str, as_attachment=True, client=None) -> str:
    if client is None:
        client = get_s3_client(path, outside=True)
    bucket, key = split_s3_path(path)
    params = {"Bucket": bucket, "Key": key}
    if as_attachment:
        filename = key.split("/")[-1]
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
    return client.generate_presigned_url("get_object", params)


def get_s3_preview_url(path: str):
    from urllib.parse import quote

    preview_url_prefix = "http://tools.bigdata.shlab.tech/preview?path="
    return preview_url_prefix + quote(path)


def get_s3_file_preview_url(path: str):
    from urllib.parse import quote

    preview_url_prefix = "http://tools.bigdata.shlab.tech/file_preview?path="
    return preview_url_prefix + quote(path)
