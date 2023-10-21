import os
import pickle
import time

from Face_Recognition import FaceRecognizer, FaceModel
from Face_Recognition import configs as face_configs
from tools.lzc.face.framework.IFaceRegMgr import IFaceRegMgr


class FaceRegMgr(IFaceRegMgr):
    def __init__(self, face_update_path, can_update=False):
        """
        构造函数

        参数:
        can_update (bool): 是否有权限更新特征库
        face_update_path (str): 特征库更新标识文件路径

        """
        self.face_update_path = face_update_path
        self.can_udpate = can_update
        self.face_model: FaceModel = None
        if can_update and os.path.exists(face_update_path):
            os.remove(face_update_path)
        self.check_and_update()

    def check_and_update(self):
        if os.path.exists(self.face_update_path):
            return False

        # 核心进程
        if self.can_udpate:
            # 生成人脸数据库
            if self.face_model is None:  # 核心进程首次加载
                self.face_model = FaceModel(database=face_configs.database, local_load=False)  # 加载新的人脸识别模型

            self.face_model.create_database(portrait=face_configs.portrait, vis=False)
            # 成功更新特征库，生成文件标识
            write_data = {"face": "running"}
            pickle.dump(write_data, open(self.face_update_path, 'wb'))
        # 非核心进程
        else:
            # 其他进程等待特征库生成
            while not os.path.exists(self.face_update_path):
                time.sleep(1)

            if self.face_model is None:
                self.face_model = FaceModel(database=face_configs.database, local_load=True)  # 加载新的人脸识别模型
            else:
                self.face_model.faceReg.load(file=face_configs.database)
                self.face_model.faceReg.update()

        return True

    def inference(self, img):
        if type(img) is str:
            return self.face_model.detect_image(img)
        else:
            return self.face_model.detect_image_ram(img)


if __name__ == "__main__":
    faceRegMgr = FaceRegMgr()

