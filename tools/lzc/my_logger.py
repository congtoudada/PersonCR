import os

from loguru import logger


def log_config(main_id):
    logger.remove()  # 避免打印到控制台
    log_path = os.path.join("logs", f"main_process{main_id}.log")
    # if os.path.exists(log_path):
    #     os.remove(log_path)
    logger.add(sink=log_path, rotation="100 MB")  # 每100MB重新写
