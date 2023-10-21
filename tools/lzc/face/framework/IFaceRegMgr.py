from abc import ABC, abstractmethod
from enum import Enum

class IFaceRegMgr(ABC):
    @abstractmethod
    def check_and_update(self):
        """
        定期调用，检查是否需要更新特征库，如果需要则更新
        """
        pass


    @abstractmethod
    def inference(self, img) -> (int, float):
        """
        推理

        参数:
        img (): 人脸图片

        返回:
        person_id (int): 人员id
        score (float): 分数
        """
        pass
