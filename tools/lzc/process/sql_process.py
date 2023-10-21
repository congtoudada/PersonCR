import os
import threading
import time
import traceback

from loguru import logger
from multiprocessing import Manager

from tools.lzc.config_tool import ConfigTool
from tools.lzc.sql.framework.SqlTool import SqlTool
from tools.lzc.sql.framework.MysqlMgr import MysqlMgr
from tools.lzc.sql.logic.Sql2Web import Sql2Web
from tools.lzc.sql.framework.SqlEnum import SqlOpEnum

"""
process_id: 自定义进程id
sqlQueue: 数据库消息队列
sqlEvent: 数据库初始化事件
escEvent: 退出事件
main_yaml: 主配置文件
"""
def sql_process(process_id, sqlQueue, sqlEvent, escEvent, main_yaml):
    pname = f'[ {os.getpid()}:sql_process {process_id} ]'

    # 使用spawn会创建全新进程，需重新配置日志
    ConfigTool.load_log_config(main_yaml, "Mysql")

    # 读取配置文件
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
    sqlMgr = MysqlMgr()
    # 连接数据库
    is_connect = sqlMgr.connect(host, port, user, pwd, database_name, pool_size)
    # 增加监听
    sql2Web = Sql2Web(web_ip, web_port)
    sqlMgr.addListener(SqlOpEnum.INSERT, True, lambda params: threading.Thread(target=sql2Web.send_renlian_alarm, args=(params['last_id'],)).start())
    sqlMgr.addListener(SqlOpEnum.INSERT, True, lambda params: threading.Thread(target=sql2Web.send_keliu_alarm, args=(params['last_id'],)).start())
    sqlMgr.addListener(SqlOpEnum.INSERT, True, lambda params: threading.Thread(target=sql2Web.send_change_format, args=(params['vid_url'],)).start())

    while not is_connect:
        time.sleep(10)  # 如果没有连接，则每隔10s重新连接
        is_connect = sqlMgr.connect(host, port, user, pwd, database_name, pool_size)

    logger.info(f'{pname} launch!')
    if sqlEvent is not None:
        sqlEvent.set()

    try:
        while True:
            if not sqlQueue.empty():  # 数据库队列不为空，则处理数据
                # logger.info(f"interface len: {len(sqlQueue)}")
                pack_data = sqlQueue.get()
                run_mode, data = SqlTool.unpack_data(packData)

                # 插入数据
                execute_result = sqlMgr.get_opContainer().getMysqlInsertOp().update(run_mode, data).execute()

                # 插入失败，数据丢弃（也可以重新插回队列）
                if not execute_result:
                    logger.error(f"{pname} 插入数据失败！数据被丢弃！")
            else:
                if escEvent.is_set():
                    logger.info(f"{pname} Exit!")
                    return
                time.sleep(sleep_time)  # 暂停，让出控制权

    except Exception as e:
        logger.error(f"{pname} {db_type}报错: {traceback.format_exc()}")


if __name__ == "__main__":

    main_yaml = ConfigTool.load_main_config("main1")
    ConfigTool.load_log_config(main_yaml)

    q = Manager().Queue()
    # 初始化SQL管理类
    sqlMgr = MysqlMgr()
    sqlMgr.connect('localhost', 3306, 'root', '123456', 'pydb')
    sql2Web = Sql2Web('127.0.0.1', 8080)
    sqlMgr.addListener(SqlOpEnum.INSERT, True, lambda params: threading.Thread(target=sql2Web.send_renlian_alarm,
                                                                               args=(params['last_id'],)).start())
    sqlMgr.addListener(SqlOpEnum.INSERT, True, lambda params: threading.Thread(target=sql2Web.send_keliu_alarm,
                                                                               args=(params['last_id'],)).start())
    sqlMgr.addListener(SqlOpEnum.INSERT, True, lambda params: threading.Thread(target=sql2Web.send_change_format,
                                                                               args=(params['vid_url'],)).start())

    data = SqlTool.pack_test_data('123', 'haha')
    q.put(data)

    time.sleep(1)  # 模拟消息传输
    packData = q.get()
    run_mode, sql_dict = SqlTool.unpack_data(packData)
    logger.info("unpack: " + run_mode.__str__() + " 表 : " + sql_dict.__len__().__str__())
    execute_result = sqlMgr.get_opContainer().getMysqlInsertOp().update(run_mode, sql_dict).execute()
    logger.info("插入结果: " + execute_result.__str__())
