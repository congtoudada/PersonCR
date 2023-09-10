import os

from loguru import logger

def log_config(main_yaml: dict):
    # 是否同时在控制台输出日志内容
    if not main_yaml.get('enable_log'):
        logger.remove()
        logger.level("CRITICAL")
        return
    elif not main_yaml.get('enable_debug'):
        logger.remove()

    log_path = os.path.join("logs", "{time:YYYY-MM-DD}_main" + str(main_yaml['main_id']) +".log")
    logger.add(sink=log_path, rotation="daily")  # 每100MB重新写
