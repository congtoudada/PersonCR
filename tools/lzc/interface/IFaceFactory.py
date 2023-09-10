from abc import ABC, abstractmethod
from enum import Enum

class IFaceFactory(ABC):
    # 加载facemodel
    @abstractmethod
    def reload(self, is_main, face_update_path, face_model=None):
        pass
