import os
import time
import traceback

import pymysql

from tools.lzc.sql.framework.SqlTool import SqlTool
from tools.lzc.sql.framework.IMySqlOp import IMysqlOp
from tools.lzc.sql.logic.MysqlOpContainer import MysqlOpContainer

from pymysqlpool import ConnectionPool
from loguru import logger

from tools.lzc.sql.framework.SqlEnum import SqlOpEnum

class MysqlMgr:
    def __init__(self):
        self.pool = None
        self.config = {}  # 数据库配置
        self.subject = {}  # 事件监听dict
        self.opContainer = MysqlOpContainer(self)  # 操作容器
        self.coder = SqlTool()

    def connect(self, host, port, user, password, database_name, pool_size=2) -> bool:
        result = False
        self.config['host'] = host
        self.config['port'] = port
        self.config['user'] = user
        self.config['password'] = password
        self.config['database_name'] = database_name
        self.config['pool_size'] = pool_size
        try:
            # 创建连接池
            self.pool = ConnectionPool(
                size=pool_size,
                name='mydb',
                host=host,
                port=port,
                user=user,
                password=password,
                db=database_name,
            )
            result = True
        except Exception as e:
            logger.error(f"pid:{os.getpid()} MySQL错误: 连接数据库失败 {traceback.format_exc()}")
        return result

    def __re_connect(self) -> bool:
        try:
            # 创建连接池
            self.pool = ConnectionPool(
                size=self.config['pool_size'],
                name='mydb',
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                db=self.config['database_name'],
            )
            return True
        except Exception as e:
            logger.error(f"pid:{os.getpid()} MySQL错误: 连接数据库失败 {traceback.format_exc()}")
            return False

    def get_opContainer(self):
        return self.opContainer

    def get_coder(self):
        return self.coder

    def execute_query(self, op: IMysqlOp) -> bool:
        pass

    # 执行sql
    # Succ:
    #   result['last_id'] 存放最后应用的主键
    #   result['vid_url'] 存放视频url
    # Fail:
    #   result['fail_info'] 存放失败信息
    def execute_modify(self, op: IMysqlOp) -> bool:
        conn = self.__get_connection()
        cursor = conn.cursor()
        is_succ = False
        result = {}

        try:
            if op.get_sql() == "":
                return False

            # 运行sql语句
            cursor.execute(op.get_sql(), op.get_sqlParams())

            # 提交修改
            conn.commit()

            # 后处理
            last_id = cursor.lastrowid  # 获取插入记录的主键ID

            # 触发成功回调函数
            event_params = {'last_id': last_id}
            event_params.update(op.get_eventParams())
            self.__notify(op.get_opEnum(), True, event_params)

            # 打印成功信息
            sql_info = op.get_sql().replace('%s', '{}').format(*op.get_sqlParams())
            logger.info(f"pid:{os.getpid()} MySQL运行SQL成功: {sql_info}")

            is_succ = True

        except pymysql.OperationalError as e:
            # 发生错误时回滚
            conn.rollback()
            logger.error(f"pid:{os.getpid()} MySQL运行SQL错误: {e}")
            self.__notify(op.get_opEnum(), False, {'fail_info': traceback.format_exc()})
            self.__re_connect()  # 尝试重新连接
        except Exception as e:
            # 发生错误时回滚
            conn.rollback()
            logger.error(f"pid:{os.getpid()} MySQL运行SQL错误: {traceback.format_exc()}")
            # 处理sql结果
            self.__notify(op.get_opEnum(), False, {'fail_info': traceback.format_exc()})
        finally:
            cursor.close()
            conn.close()

        return is_succ

    def addListener(self, op_enum: SqlOpEnum, is_succ: bool, func: callable(dict)) -> None:
        if not self.subject.__contains__(op_enum):
            self.subject[op_enum] = {
                'success': [],
                'fail': []
            }

        self.subject[op_enum]['success' if is_succ else 'fail'].append(func)

    def __notify(self, op_enum: SqlOpEnum, is_succ: bool, result: dict) -> None:
        if self.subject == {} or not self.subject.__contains__(op_enum):
            return

        for item in self.subject[op_enum]['success' if is_succ else 'fail']:
            item(result)

    def __get_connection(self):
        if self.pool is None:
            logger.error(f"pid:{os.getpid()} MySQL错误: 没有数据库连接")
            return None
        else:
            return self.pool.get_connection()


if __name__ == "__main__":
    # 插入测试
    factory = MysqlMgr()
    factory.connect('localhost', 3306, 'root', '123456', 'pydb')
    factory.addListener(SqlOpEnum.INSERT, True, lambda result: logger.info(f"插入成功时触发: {result['last_id']}"))

    # factory.getOpContainer().getxxxOp().update(...).execute()
    start_time = time.time()
    factory.get_opContainer().getMysqlInsertOp().update(-1, {'password': '2023', 'description': '测试pymysql'}).execute()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序运行耗时：{execution_time}秒")
