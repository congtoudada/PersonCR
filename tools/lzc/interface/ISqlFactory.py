from abc import ABC, abstractmethod
from enum import Enum

class SqlEventEnum(Enum):
    ON_SUCC_INSERT = 0  # 当插入成功时调用
    ON_FAIL_INSERT = 1
    ON_SUCC_DELETE = 2
    ON_FAIL_DELETE = 3
    ON_SUCC_UPDATE = 4
    ON_FAIL_UPDATE = 5
    ON_SUCC_QUERY = 6
    ON_FAIL_QUERY = 7

class SqlOpEnum(Enum):
    INSERT = 0
    DELETE = 1
    UPDATE = 2
    QUERY = 3

class ISqlFactory(ABC):
    # 数据库连接
    @abstractmethod
    def connect(self, host, port, user, password, database_name, pool_size) -> bool:
        pass

    # 得到一个查询类型
    @abstractmethod
    def createSqlOp(self, sqlOpEnum: SqlOpEnum, run_mode: int, key_values: dict) -> dict:
        pass

    # 执行查询，返回主键
    @abstractmethod
    def execute(self, sqlOp: dict) -> bool:
        pass

    # 增加监听事件
    @abstractmethod
    def addListener(self, sqlEvent: SqlEventEnum, func: callable(dict)) -> None:
        pass