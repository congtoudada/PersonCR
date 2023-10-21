from abc import ABC, abstractmethod
from tools.lzc.sql.framework.SqlEnum import SqlOpEnum


class IMysqlOp(ABC):
    # 得到Op类型
    @abstractmethod
    def get_opEnum(self) -> SqlOpEnum:
        pass

    # 得到Sql
    @abstractmethod
    def get_sql(self) -> str:
        pass

    # 得到sql数据
    @abstractmethod
    def get_sqlParams(self) -> dict:
        pass

    # 得到后处理数据
    @abstractmethod
    def get_eventParams(self) -> dict:
        pass