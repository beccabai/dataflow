import os
import re

def get_user_name():
    """
    get user name from environment variable USER_NAME
    assert the user name is not empty
    Returns:
        str: user name
    """
    try:
        user_name = os.environ["USER_NAME"]
    except Exception as e:
        print(f"get user name failed: {e}")
        return None
    assert len(user_name.split()) > 0
    return user_name

def get_ray_head_ip_from_dir(ray_info_dir):
    """
    Get the ip of the ray head node
    by listing all dir in ray_info_dir and assert only one dir and the dir name is ip
    """
    def is_ip(input_str):
        return re.match(r"\d+\.\d+\.\d+\.\d+", input_str)

    # list all dir in ray_info_dir like "/fs-computility/llm/shared/qiujiantao/ray_server_infos/10.18.0.55_6379"
    dir_list = [f for f in os.listdir(ray_info_dir) if os.path.isdir(os.path.join(ray_info_dir, f)) and is_ip(f)]
    # assert only one dir and the dir name is ip
    assert len(dir_list) == 1
    return dir_list[0].split("_")[0]
