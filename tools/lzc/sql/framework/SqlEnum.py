from enum import Enum

class SqlOpEnum(Enum):
    INSERT = 0
    DELETE = 1
    UPDATE = 2
    QUERY = 3


# class SqlEventEnum(Enum):
#     ON_SUCC_INSERT = 0  # 当插入成功时调用
#     ON_FAIL_INSERT = 1
#     ON_SUCC_DELETE = 2
#     ON_FAIL_DELETE = 3
#     ON_SUCC_UPDATE = 4
#     ON_FAIL_UPDATE = 5
#     ON_SUCC_QUERY = 6
#     ON_FAIL_QUERY = 7
