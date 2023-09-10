import asyncio
import os
import time

from loguru import logger
from multiprocessing import Process, Manager

from tools.lzc import SqlHelper
from tools.lzc.concrete.MysqlFactory import MysqlFactory
from tools.lzc.concrete.Sql2Web import Sql2Web
from tools.lzc.interface.ISqlFactory import ISqlFactory, SqlOpEnum, SqlEventEnum
from tools.lzc.my_logger import log_config

"""
process_id: 自定义进程id
sqlQueue: 数据库消息队列
sqlEvent: 数据库初始化事件
escEvent: 退出事件
main_yaml: 主配置文件
"""
def sql_process(process_id, sqlQueue, sqlEvent, escEvent, main_yaml):
    pname = f'[ {os.getpid()}:sql_process {process_id} ]'
    logger.info(f'{pname} launch!')

    # 使用spawn会创建全新进程，需重新配置日志
    log_config(main_yaml)

    db_type = main_yaml['database']['type']
    sleep_time = main_yaml['database']['sleep']
    host = main_yaml['database']['host']
    port = main_yaml['database']['port']
    user = main_yaml['database']['user']
    pwd = main_yaml['database']['password']
    database_name = main_yaml['database']['database_name']
    pool_size = main_yaml['database']['pool_size']
    web_ip = main_yaml['database']['web_ip']
    web_port = main_yaml['database']['web_port']

    # 初始化SQL工厂
    factory: ISqlFactory = MysqlFactory()
    # 连接数据库
    result = factory.connect(host, port, user, pwd, database_name, pool_size)
    # 数据库帮助类
    sqlHelper = SqlHelper()
    # 增加监听
    sql2Web = Sql2Web(web_ip, web_port)
    factory.addListener(SqlEventEnum.ON_SUCC_INSERT, lambda params: sql2Web.send_renlian_alarm(params['last_id']))
    factory.addListener(SqlEventEnum.ON_SUCC_INSERT, lambda params: sql2Web.send_keliu_alarm(params['last_id']))
    factory.addListener(SqlEventEnum.ON_SUCC_INSERT, lambda params: sql2Web.send_change_format(params['vid_url']))

    while not result:
        time.sleep(10)  # 如果没有连接，则每隔10s重新连接
        result = factory.connect(host, port, user, pwd, database_name, pool_size)

    if sqlEvent is not None:
        sqlEvent.set()

    try:
        while True:
            if not sqlQueue.empty():  # 数据库队列不为空，则处理数据
                # logger.info(f"sql len: {len(sqlQueue)}")
                packData = sqlQueue.get()
                table_name, sql_dict = sqlHelper.unpackData(packData)
                sqlItem = factory.createSqlOp(SqlOpEnum.INSERT, table_name, sql_dict)
                re = factory.execute(sqlItem)
                if not re:
                    logger.error(f"{pname} 插入数据失败！数据被丢弃！")
            else:
                if escEvent.is_set():
                    logger.info(f"{pname} Exit!")
                    return
                time.sleep(sleep_time)  # 暂停，让出控制权
    except Exception as e:
        logger.error(f"{pname} {db_type}报错: {e}")


if __name__ == "__main__":
    q = Manager().Queue()
    # 初始化SQL工厂
    factory: ISqlFactory = MysqlFactory()
    factory.connect('localhost', 3306, 'root', '123456', 'pydb')
    sqlHelper = SqlHelper()
    sql2Web = Sql2Web('127.0.0.1', 8080)
    factory.addListener(SqlEventEnum.ON_SUCC_INSERT, lambda params: asyncio.run(sql2Web.send_renlian_alarm(params['last_id'])))
    factory.addListener(SqlEventEnum.ON_SUCC_INSERT, lambda params: asyncio.run(sql2Web.send_keliu_alarm(params['last_id'])))
    factory.addListener(SqlEventEnum.ON_SUCC_INSERT, lambda params: asyncio.run(sql2Web.send_change_format(params['vid_url'])))

    data = sqlHelper.packTestData('123', 'haha')
    q.put(data)

    time.sleep(1)  # 模拟消息传输
    packData = q.get()
    run_mode, sql_dict = sqlHelper.unpackData(packData)
    logger.info("unpack: " + run_mode.__str__() + "表 : " + sql_dict.__len__().__str__())
    item = factory.createSqlOp(SqlOpEnum.INSERT, run_mode, sql_dict)
    re = factory.execute(item)
    logger.info("插入结果: " + re.__str__())
