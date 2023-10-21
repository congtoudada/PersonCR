# -*-coding: utf-8 -*-
import torch
import os

root = os.path.dirname(os.path.dirname(__file__))

det_thresh = 0.4  # 人脸检测阈值，小于该阈值的检测框会被剔除
rec_thresh = 0.3  # 人脸识别阈值，小于该阈值的人脸识别结果为unknown，表示未知
# 人脸检测模型,目前支持RFB和MTCNN人脸检测
DETECTOR = {
    # "net_name": "RFB",
    "net_name": "MTCNN",
}

# 人脸识别(特征提取)模型配置文件，目前支持resnet50,resnet18和mobilenet_v2模型
FEATURE = {
    "net_name": "resnet50",
    # "net_name": "resnet18",
    # "net_name": "mobilenet_v2",
    "input_size": (112, 112),
    "embedding_size": 512
}

# 人脸数据库图像路径,用于注册人脸
portrait = "./assets/face/dataset"
# 人脸数据库特征路径database(注册人脸后生成的特征文件)
database = os.path.join(os.path.dirname(portrait), "database-{}.json".format(FEATURE['net_name']))

# 运行设备
# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
