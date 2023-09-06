import os
import pickle
import time

from loguru import logger
from facenet.facenet import *

# Facenet版本
class FaceHelper:
    def __init__(self, is_trt=False, is_force_update=False):
        self.name = f"[ FaceHelper:{os.getpid()} ]"
        self.facenet = Facenet(is_trt)
        self.facenet.load_encodings(is_force_update) # 自动更新特征库

    # 传入人脸图片路径，进行人脸识别（二次匹配，两次都为陌生人才是陌生人）
    # 如果存在匹配项，返回人的id
    # 如果无匹配项，返回id=1
    def predict(self, img_path, is_vis=False):
        datasets_ids, dataset_encodings = self.load_encodings()  # 取得特征库编码向量 ids, encodings
        person_id = self._predict_one(img_path, datasets_ids, dataset_encodings, is_vis=is_vis)
        if person_id == 1:
            second_path = img_path.replace(".jpg", "1.jpg")
            if os.path.exists(second_path):  # 候选图存在，使用候选图二次匹配
                person_id = self._predict_one(second_path, datasets_ids, dataset_encodings, is_vis=is_vis)

        return person_id

    def _predict_one(self, img_path, datasets_ids, dataset_encodings, is_vis=False):
        return self.facenet.predict_one(img_path, datasets_ids, dataset_encodings, is_vis=is_vis)

    # 缓存文件校验与获取
    def load_encodings(self, is_force_update=False):
        return self.facenet.load_encodings(is_force_update)


if __name__ == "__main__":
    faceHelper = FaceHelper(is_force_update=True)
    test_dir = r"C:\Users\13220\Desktop\test"
    # test_dir = r"E:\Practice\AI\School\MOT\Bytetrack\facenet\lfw\Yang"
    for filename in os.listdir(test_dir):
        faceHelper.predict(os.path.join(test_dir, filename), is_vis=True)
