# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-04-05 14:09:34
# --------------------------------------------------------
"""
import os
import argparse
from Face_Recognition.configs import configs
from Face_Recognition.core.face_recognizer import FaceRecognizer


def parse_opt():
    # portrait = "./data/database/portrait"  # 人脸肖像图像路径
    # database = os.path.join(os.path.dirname(image_dir), "database.json")
    portrait = configs.portrait  # 人脸肖像图像路径
    database = configs.database  # 存储人脸数据库特征路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--portrait', type=str, default=portrait, help='人脸数据库目录')
    parser.add_argument('--database', type=str, default=database, help='存储人脸数据库特征路径')
    opt = parser.parse_args()
    print(opt)
    return opt


def register():
    """
    注册人脸，生成人脸数据库
    portrait：人脸数据库图片目录，要求如下：
              (1) 图片按照[ID-XXXX.jpg]命名,如:张三-image.jpg，作为人脸识别的底图
              (2) 人脸肖像照片要求五官清晰且正脸的照片，不能出现多个人脸的情况
    @return:
    """
    # opt = parse_opt()
    portrait = configs.portrait  # 人脸肖像图像路径
    database = configs.database  # 存储人脸数据库特征路径

    fr = FaceRecognizer(database=database)
    # 生成人脸数据库
    fr.create_database(portrait=portrait, vis=False)
    # 测试人脸识别效果
    # fr.detect_image_dir(test_dir, vis=True)

def my_register(portrait, database, local_load=False):
    fr = FaceRecognizer(database=database, local_load=local_load)
    # 生成人脸数据库
    fr.create_database(portrait=portrait, vis=False)

if __name__ == "__main__":
    register()
