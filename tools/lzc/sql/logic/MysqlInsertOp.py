import io
import os

from loguru import logger

from tools.lzc.sql.framework.IMySqlOp import IMysqlOp
from tools.lzc.sql.framework.SqlEnum import SqlOpEnum


class MysqlInsertOp(IMysqlOp):
    # sqlOp:
    #   ['sqlOpEnum'] = SqlOpEnum
    #   ['interface'] = str
    #   ['params'] = list
    #   ['other'] = dict
    def __init__(self, op_container):
        self.op_container = op_container
        self.op_enum = SqlOpEnum.INSERT
        self.sql = ""
        self.sqlParams = {}
        self.eventParams = {}

    def _reset(self):
        self.sql = ""
        self.sqlParams.clear()
        self.eventParams.clear()

    # 更新数据
    def update(self, run_mode: int, sql_dict: dict):
        # 重置状态
        self._reset()
        table_name = self.op_container.tables[run_mode]
        # 生成插入sql（这部分代码比较固定，可以根据需要写到一个工具类里面噢，由于项目比较单一就不做展开）
        sqlKey = io.StringIO()
        sqlValue = io.StringIO()
        sqlKey.write(f'({table_name}_id,')
        sqlValue.write('(null,')  # 主键自增
        params = []
        len = sql_dict.__len__()
        i = 0
        for key in sql_dict.keys():
            i += 1
            sqlKey.write(key)
            sqlValue.write('%s')
            params.append(sql_dict[key].__str__())
            if i < len:
                sqlKey.write(',')
                sqlValue.write(',')
            else:
                sqlKey.write(')')
                sqlValue.write(')')

        self.sql = f"insert into {table_name} {sqlKey.getvalue()} values {sqlValue.getvalue()}"
        logger.info(f"pid:{os.getpid()} MySQL创建Sql语句: {self.sql}")

        self.sqlParams = params
        self.eventParams['vid_url'] = sql_dict.get('record_video_url', None)
        return self

    def execute(self) -> bool:
        return self.op_container.sql_factory.execute_modify(self)

    def get_opEnum(self):
        return self.op_enum

    def get_sql(self):
        return self.sql

    def get_sqlParams(self):
        return self.sqlParams

    def get_eventParams(self):
        return self.eventParams


