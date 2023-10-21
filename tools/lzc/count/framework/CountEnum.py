import enum
from enum import Enum


# Valid: 进入检测区域且有效
# Invalid: 进入检测区域但无效
# Hidden: 进入检测区域后消失
# Dying: 离开缓冲区域
# Dead: 离开缓冲区域一段时间
class CountItemStateEnum(Enum):
    Valid = 0
    Invalid = 1
    Dying = 2
    Dead = 3


# In: 进入方向所在区域
# Out: 离开方向所在区域
class ZoneEnum(Enum):
    Null = 0
    In = 1
    Out = 2

# NoSend: 不可发送sql请求
# CanSend: 可发送sql请求但还未发送
# Sended: 已经发送sql请求
class FaceStateEnum(Enum):
    NoSend = 0
    CanSend = 1
    Sended = 2
