import os
import time

from loguru import logger
from multiprocessing import Process, current_process

from tools.lzc.sql_helper import MySQLHelpler, KeliuSQLObj, RenlianSQLObj
from tools.lzc.my_logger import log_config

# 测试代码
# time.sleep(1) # 模拟进程启动
# print(f"{process_name} 启动完毕")
# if sqlEvent is not None:
#     sqlEvent.set()
# return
def sql_process(process_id, p_qsql, sqlEvent, esc_event, main_yaml):
    pname = f'[ {os.getpid()}:sql_process {process_id} ]'
    logger.info(f'{pname} launch!')
    print(f'{pname} launch!')
    log_config(main_id=main_yaml['main_id'])

    sleep_time = main_yaml['mysql']['sleep']
    is_sql = main_yaml['mysql']['is_sql']
    is_debug = main_yaml['is_debug']
    host = main_yaml['mysql']['host']
    port = main_yaml['mysql']['port']
    user = main_yaml['mysql']['user']
    pwd = main_yaml['mysql']['password']
    database = main_yaml['mysql']['database']
    web_ip = main_yaml['mysql']['web_ip']
    web_port = main_yaml['mysql']['web_port']

    sqlHelper = None
    try:
        if is_sql:
            sqlHelper = MySQLHelpler({"host": host, "port": port, "user": user,
                                      "password": pwd.__str__(), "database": database,
                                      "web_ip": web_ip, "web_port": web_port},
                                     is_debug=is_debug)
    except Exception as e:
        if is_debug:
            logger.error(f"{pname} init fail: {e}")

    if sqlEvent is not None:
        sqlEvent.set()
    refresh_flag = 0
    while True:
        if not p_qsql.empty():
            # logger.info(f"sql len: {len(p_qsql)}")
            # 数据库没连接，就尝试连接
            if is_sql and not sqlHelper.is_connect:
                print(f"pid:{os.getpid()} 没有数据库连接，尝试连接")
                result = sqlHelper.connect_mysql(sqlHelper.config)
                if not result:  # 连接失败
                    time.sleep(3)  # 暂停5s再试
                    continue
            elif is_sql and not sqlHelper.db.open:
                print(f"pid:{os.getpid()} 数据库连接断开，尝试连接")
                result = sqlHelper.connect_mysql(sqlHelper.config)
                if not result:  # 连接失败
                    time.sleep(3)  # 暂停5s再试
                    continue

            # 连接成功才读取数据
            sql_dict = p_qsql.get()
            # if is_debug:
            # print(f'recv sql_dict: {sql_dict}')
            if sqlHelper is not None:
                run_mode = sql_dict['run_mode']
                item = None
                if run_mode == 0:  # 客流
                    item = KeliuSQLObj(record_time=sql_dict['record_time'],
                                       flow_cam_id=sql_dict['flow_cam_id'],
                                       record_status=sql_dict['record_status'],
                                       record_num=sql_dict['record_num'],
                                       record_photo_url=sql_dict['record_photo_url'],
                                       is_warning=sql_dict['is_warning'],
                                       record_video_url=sql_dict['record_video_url'])
                elif run_mode == 1:  # 人脸
                    item = RenlianSQLObj(record_time=sql_dict['record_time'],
                                         recognize_cam_id=sql_dict['recognize_cam_id'],
                                         record_status=sql_dict['record_status'],
                                         record_num=sql_dict['record_num'],
                                         record_photo_url=sql_dict['record_photo_url'],
                                         personnel_id=sql_dict['personnel_id'],
                                         is_warning=sql_dict['is_warning'],
                                         record_video_url=sql_dict['record_video_url'])

                if item is not None:
                    sqlHelper.execute_sql(item)
                else:
                    logger.error("Error: sql item is null!")
        else:
            if esc_event.is_set():
                logger.info(f"{pname} Exit!")
                print(f"{pname} Exit!")
                return
            time.sleep(sleep_time)  # 暂停1s，让出控制权
            refresh_flag += 1
            if refresh_flag % 10800 == 0:
                refresh_flag = 0
                sqlHelper.run()


def start_sql_process(p_qsql, sqlEvent, esc_event, main_yaml):
    psql = Process(target=sql_process, name="sql_process",
                   args=("sql", p_qsql, sqlEvent, esc_event, main_yaml))
    psql.start()
    return psql
