import io
import os
import time

import pymysql

from tools.lzc.interface.ISqlFactory import ISqlFactory, SqlEventEnum, SqlOpEnum
from pymysqlpool import ConnectionPool
from loguru import logger

class MysqlFactory(ISqlFactory):
    def __init__(self):
        self.pool = None
        self.config = {}
        self.subject = {}
        self.sqlOp = {}
        self.tables = {-1: 'user', 0: 'flow_record', 1: 'recognize_record'}

    def connect(self, host, port, user, password, database_name, pool_size=5) -> bool:
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
            logger.error(f"pid:{os.getpid()} MySQL错误: 连接数据库失败 {e}")
        return result

    # sqlOp:
    #   ['sqlOpEnum'] = SqlOpEnum
    #   ['sql'] = str
    #   ['params'] = list
    #   ['other'] = dict
    def createSqlOp(self, sqlOpEnum: SqlOpEnum, run_mode: int, key_values: dict) -> dict:
        self.sqlOp = {'sqlOpEnum': sqlOpEnum}
        table_name = self.tables[run_mode]
        sqlKey = io.StringIO()
        sqlValue = io.StringIO()

        sqlKey.write(f'({table_name}_id,')
        sqlValue.write('(null,')  # 主键自增
        params = []
        len = key_values.__len__()
        i = 0
        for key in key_values.keys():
            i += 1
            sqlKey.write(key)
            sqlValue.write('%s')
            params.append(key_values[key].__str__())
            if i < len:
                sqlKey.write(',')
                sqlValue.write(',')
            else:
                sqlKey.write(')')
                sqlValue.write(')')

        sql = ""
        # 根据项目需求实现
        if sqlOpEnum == SqlOpEnum.INSERT:
            sql = f"insert into {table_name} {sqlKey.getvalue()} values {sqlValue.getvalue()}"
            logger.info(f"pid:{os.getpid()} MySQL创建Sql语句: {sql}")
        else:
            logger.warning(f"pid:{os.getpid()} MySQL错误: 没有实现该类型的操作 {SqlOpEnum.name}")

        self.sqlOp['sql'] = sql
        self.sqlOp['params'] = params
        self.sqlOp['other'] = {'vid_url': key_values.get('record_video_url', None)}
        return self.sqlOp

    # Succ:
    #   Query: result['query_result'] 存放查询结果
    #   Insert,Delete,Update:
    #       result['last_id'] 存放最后应用的主键
    #       result['vid_url'] 存放视频url
    # Fail:
    #   result['fail_info'] 存放失败信息
    def execute(self, sqlOp: dict) -> bool:
        if self.pool is None:
            logger.error(f"pid:{os.getpid()} MySQL错误: 没有数据库连接")
            return False

        conn = self.pool.get_connection()
        cursor = conn.cursor()
        sqlOpEnum: SqlOpEnum = sqlOp['sqlOpEnum']
        result = {}
        isSucc = False
        try:
            if sqlOp['sql'] == "":
                return False

            # 运行sql语句
            cursor.execute(sqlOp['sql'], sqlOp['params'])

            # 处理sql结果
            if sqlOpEnum == SqlOpEnum.QUERY:
                result['query_result'] = cursor.fetchall()
                self._notify(SqlEventEnum.ON_SUCC_QUERY, result)
            else:
                conn.commit()  # 提交修改
                last_id = cursor.lastrowid # 获取插入记录的主键ID
                result['last_id'] = last_id
                result['vid_url'] = sqlOp['other']['vid_url']
                if sqlOpEnum == SqlOpEnum.INSERT:
                    self._notify(SqlEventEnum.ON_SUCC_INSERT, result)
                elif sqlOpEnum == SqlOpEnum.DELETE:
                    self._notify(SqlEventEnum.ON_SUCC_DELETE, result)
                else:
                    self._notify(SqlEventEnum.ON_SUCC_UPDATE, result)

            isSucc = True
            sql_info = sqlOp['sql'].replace('%s', '{}').format(*sqlOp['params'])
            logger.info(f"pid:{os.getpid()} MySQL运行SQL成功: {sql_info}")
        except pymysql.OperationalError as e:
            # 发生错误时回滚
            conn.rollback()
            logger.error(f"pid:{os.getpid()} MySQL运行SQL错误: {e}")
            self._re_connect()  # 尝试重新连接

        except Exception as e:
            # 发生错误时回滚
            conn.rollback()
            logger.error(f"pid:{os.getpid()} MySQL运行SQL错误: {e}")

            # 处理sql结果
            result['fail_info'] = e.__str__()
            if sqlOpEnum == SqlOpEnum.QUERY:
                self._notify(SqlEventEnum.ON_FAIL_QUERY, result)
            else:
                if sqlOpEnum == SqlOpEnum.INSERT:
                    self._notify(SqlEventEnum.ON_FAIL_INSERT, result)
                elif sqlOpEnum == SqlOpEnum.DELETE:
                    self._notify(SqlEventEnum.ON_FAIL_DELETE, result)
                else:
                    self._notify(SqlEventEnum.ON_FAIL_UPDATE, result)
        finally:
            cursor.close()
            conn.close()

        return isSucc

    def addListener(self, sqlEvent: SqlEventEnum, func: callable(dict)) -> None:
        if not self.subject.__contains__(sqlEvent):
            self.subject[sqlEvent] = []
        self.subject[sqlEvent].append(func)

    def _notify(self, sqlEvent: SqlEventEnum, result):
        if self.subject == {} or self.subject[sqlEvent] is None:
            return

        for item in self.subject[sqlEvent]:
            item(result)

    def _re_connect(self):
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

if __name__ == "__main__":
    # 插入测试
    factory: ISqlFactory = MysqlFactory()
    factory.connect('localhost', 3306, 'root', '123456', 'pydb')
    factory.addListener(SqlEventEnum.ON_SUCC_INSERT, lambda result: logger.info(f"插入成功时触发: {result['last_id']}"))

    start_time = time.time()
    sqlOp = factory.createSqlOp(SqlOpEnum.INSERT, -1, {'password': '2023', 'description': '测试pymysql'})
    re = factory.execute(sqlOp)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序运行耗时：{execution_time}秒")





