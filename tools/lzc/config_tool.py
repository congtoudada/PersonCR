import os
import time
import yaml
from loguru import logger


class ConfigTool:
    @staticmethod
    def _update(template_dict: dict, override_dict: dict):
        # 更新模板参数
        for key in template_dict.keys():
            if override_dict.__contains__(key):
                # 如果value是字典，需要递归
                if type(template_dict[key]) is dict:
                    ConfigTool._update(template_dict[key], override_dict[key])
                else:
                    template_dict[key] = override_dict[key]

    @staticmethod
    def _addNew(template_dict: dict, override_dict: dict):
        # 添加新参数
        for key in override_dict.keys():
            if not template_dict.__contains__(key):
                template_dict[key] = override_dict[key]
    @staticmethod
    def load_main_config(main_file: str) -> dict:
        """
        加载主配置文件

        参数:
        main_file (str): main函数文件名

        返回:
        dict
        """
        # 加载yaml主配置
        main_yaml_template = ConfigTool.read_yaml(file=f"exps/custom/template/main_template.yaml")
        main_yaml = ConfigTool.read_yaml(file=main_file)
        # 更新
        ConfigTool._update(main_yaml_template, main_yaml)
        ConfigTool._addNew(main_yaml_template, main_yaml)
        return main_yaml_template

    @staticmethod
    def load_cam_config(cam_file: str) -> dict:
        """
        加载相机配置文件

        参数:
        run_mode: 0-客流 1-人脸
        main_file (str): main函数文件名

        返回:
        dict
        """
        cam_yaml = ConfigTool.read_yaml(file=cam_file)
        # 加载yaml主配置
        run_mode = cam_yaml['run_mode']
        if run_mode == 0:
            cam_yaml_template = ConfigTool.read_yaml(file=f"exps/custom/template/keliu_template.yaml")
        else:
            cam_yaml_template = ConfigTool.read_yaml(file=f"exps/custom/template/renlian_template.yaml")

        # 更新
        ConfigTool._update(cam_yaml_template, cam_yaml)
        ConfigTool._addNew(cam_yaml_template, cam_yaml)
        return cam_yaml_template

    #   ①
    @staticmethod
    def load_log_config(main_yaml: dict, tag: str = None, is_clean=False):
        """
        加载日志配置

        参数:
        main_yaml (dict): main函数配置文件
        is_clean (bool): 是否执行定期清理逻辑
        tag (str): 文件标识名，针对不同模块可以给不同的名称，生成不同的日志文件

        返回:
        None
        """
        if is_clean:
            localtime = time.localtime()
            begin_time_str = time.strftime('%Y-%m-%d', localtime)
            # 每2个月清理1个月的日志
            now_month = int(begin_time_str.split('-')[1])
            # 使用os.walk()遍历文件夹
            for root, dirs, files in os.walk("logs"):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    month = int(file_path.split('-')[1])

                    diff = now_month - month
                    if diff < 0:
                        diff = now_month + 12 - month

                    if diff >= 2:
                        os.remove(file_path)

        # 是否同时在控制台输出日志内容
        if not main_yaml.get('enable_log'):
            logger.remove()
            logger.level("CRITICAL")
            return
        elif not main_yaml.get('enable_debug'):
            logger.remove()

        if tag is None:
            log_path = os.path.join("logs", "{time:YYYY-MM-DD}_main" + str(main_yaml['main_id']) + ".log")
        else:
            log_path = os.path.join("logs", "{time:YYYY-MM-DD}_" + f"{tag}{main_yaml['main_id']}" + ".log")

        logger.add(sink=log_path, rotation="daily")  # 每100MB重新写

    # 读取Yaml文件
    @staticmethod
    def read_yaml(file, encoding='utf-8'):
        with open(file, encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    # 写入Yaml文件
    @staticmethod
    def write_yaml(file, wtdata, encoding='utf-8'):
        with open(file, encoding=encoding, mode='w') as f:
            yaml.dump(wtdata, stream=f, allow_unicode=True)
