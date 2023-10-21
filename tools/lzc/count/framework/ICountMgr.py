from abc import ABC, abstractmethod

class ICountMgr(ABC):
    @abstractmethod
    def update(self, im, tlwhs, ids, scores, now, frame_id):
        """
        每帧调用，计数核心逻辑
        """
        pass

    @abstractmethod
    def global_update(self, im, now, frame_id):
        """
        每帧最后调用，资源管理
        """
        pass

    @abstractmethod
    def get_container(self) -> dict:
        pass

    @abstractmethod
    def draw(self, im):
        """
        每帧调用(最好在update之后)，绘制可视化信息
        """
        pass

    @abstractmethod
    def debug_info(self) -> str:
        """
        返回CountMgr状态信息
        """
        pass

    @abstractmethod
    def print_params_info(self):
        """
        返回CountMgr参数信息
        """
        pass
