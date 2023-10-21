from tools.lzc.sql.logic.MysqlInsertOp import MysqlInsertOp
from tools.lzc.sql.framework.SqlEnum import SqlOpEnum

class MysqlOpContainer:
    def __init__(self, sql_factory):
        self.sql_factory = sql_factory
        self.tables = {-1: 'user', 0: 'flow_record', 1: 'recognize_record'}
        self.op_dict = {
            SqlOpEnum.INSERT: MysqlInsertOp(self)
        }

    def getMysqlInsertOp(self) -> MysqlInsertOp:
        return self.op_dict[SqlOpEnum.INSERT]