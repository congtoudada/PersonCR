import os
import threading
import pymysql
import time
import io

import requests
from loguru import logger
import urllib.request


# config:
#   host
#   user
#   password
#   database
class MySQLHelpler(threading.Thread):
    def __init__(self, config, is_debug=False):
        self.db = None
        self.config = config
        self.connect_mysql(config)
        self.last_time = time.time()
        self.inner_refresh_time = 60 * 60 * 2  # 内部刷新周期：1h
        # self.outer_refresh_time = 60 * 60 * 4 # 外部刷新周期：3h
        self.is_debug = is_debug
        self.is_connect = False
        self.web_ip = config['web_ip']
        self.web_port = config['web_port']

    def connect_mysql(self, config):
        logger.info(f"pid:{os.getpid()} 正在连接数据库:{config['host']}...")
        try:
            self.db = pymysql.connect(host=config['host'],
                                      port=config['port'],
                                      user=config['user'],
                                      password=config['password'],
                                      database=config['database'],
                                      charset='utf8')
            logger.info(f"pid:{os.getpid()} MySQL启动完成!")
            print(f"pid:{os.getpid()} MySQL启动完成!")
            self.is_connect = True
            return True
        except Exception as e:
            logger.info(f"pid:{os.getpid()} MySQL启动失败: {e}")
            return False

    def get_outer_refresh_time(self):
        return self.outer_refresh_time

    def execute_sql(self, SQLObj):
        # 维持连接
        self.refresh(self.inner_refresh_time)

        if not self.db.open:
            logger.error(f"pid:{os.getpid()} db is None, Can't operate!")
            return
        if SQLObj is None:
            logger.error(f"pid:{os.getpid()} No SQLObj, Can't execute SQL!")
            return

        vid_url = SQLObj.sqlDict['record_video_url']
        # 如果是人脸对象，就取人id
        personnel_id = 0
        cam_id = 1
        if isinstance(SQLObj, RenlianSQLObj):
            personnel_id = SQLObj.sqlDict['personnel_id']
            cam_id = SQLObj.sqlDict['recognize_cam_id']

        sqlKey = io.StringIO()
        sqlValue = io.StringIO()
        sqlKey.write(f'({SQLObj.table_name}_id,')
        sqlValue.write('(null,')  # 主键自增
        params = []
        sqlDict = SQLObj.get_dict()
        len = sqlDict.__len__()
        i = 0
        for key in sqlDict.keys():
            i += 1
            sqlKey.write(key)
            sqlValue.write('%s')
            params.append(sqlDict[key].__str__())
            if i < len:
                sqlKey.write(',')
                sqlValue.write(',')
            else:
                sqlKey.write(')')
                sqlValue.write(')')

        sql = f"insert into {SQLObj.table_name}{sqlKey.getvalue()} values {sqlValue.getvalue()}"
        # print(f"执行SQL: {sql}")

        # 执行SQL
        mysql_item = MySQLItem(self.db, sql, params, is_debug=self.is_debug,
                               webip=self.web_ip, webport=self.web_port, personnel_id=personnel_id, cam_id=cam_id,
                               vid_url=vid_url)
        mysql_item.run()

    # 定期执行，执行查询，维持连接
    def refresh(self, update_time):
        # if not self.db.open:
        #     logger.info('数据库连接失效，尝试重新连接: ')
        #     self.connect_mysql(self.config)
        now = time.time()
        if now - self.last_time > update_time:
            self.last_time = now
            self.run()

    # 执行一次空查询
    def run(self):
        cursor = self.db.cursor()
        try:
            sql = "select * from personnel where personnel_id = 1"
            cursor.execute(sql)
            logger.info(f"pid:{os.getpid()} 刷新数据库")
            # cursor.fetchall()
        except Exception:
            logger.error(f"pid:{os.getpid()} 查询失败")
        finally:
            cursor.close()


class MySQLItem(threading.Thread):
    def __init__(self, db, sql, params, is_debug=False, webip="", webport="", personnel_id=0, cam_id=1, vid_url=""):
        threading.Thread.__init__(self)
        self.db = db
        self.sql = sql
        self.params = params
        self.is_debug = is_debug
        self.webip = webip
        self.webport = webport
        self.personnel_id = personnel_id
        self.cam_id = cam_id
        self.vid_url = vid_url

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        cursor = self.db.cursor()
        try:
            # 运行sql语句
            cursor.execute(self.sql, self.params)
            # 修改
            self.db.commit()

            # 获取插入记录的主键ID
            last_id = cursor.lastrowid
            # print("Inserted ID:", last_id)

            if self.is_debug:
                # 提示成功信息
                logger.info(f"pid:{os.getpid()} 插入数据成功: {self.params}")

            # 发送消息给后端
            try:
                if self.personnel_id != 0: # 人脸
                    # url = f"http://{self.webip}:{self.webport}/common/alarmPersonnelId?PersonnelId={self.personnel_id}&CameraId={self.cam_id}"
                    # urllib.request.urlopen(url)
                    # # print(f"pid:{os.getpid()} 发送url: {url}")
                    # logger.info(f"pid:{os.getpid()} 发送url: {url}")

                    url = f"http://{self.webip}:{self.webport}/recognizeRecordEntity/setAlramByRecordId?recordId={last_id}"
                    urllib.request.urlopen(url)
                    # print(f"pid:{os.getpid()} 发送url: {url} id:{last_id}")
                    logger.info(f"pid:{os.getpid()} 发送url: {url} id:{last_id}")
                else: # 客流
                    url = f"http://{self.webip}:{self.webport}/flowRecordEntity/setAlramByRecordId?recordId={last_id}"
                    urllib.request.urlopen(url)
                    # print(f"pid:{os.getpid()} 发送url: {url} id:{last_id}")
                    logger.info(f"pid:{os.getpid()} 发送url: {url} id:{last_id}")
                    # if self.is_debug:
                    #     data = response.read().decode('utf-8')
                    #     logger.info(f"web response: {data}")
            except Exception as e:
                logger.error(f"pid:{os.getpid()} Not found valid ip or port: {self.webip}:{self.webport}")
                logger.error(e)

            try:
                # 发送后端，要求视频转码
                url = f"http://{self.webip}:9092/convert_video"  # 替换为实际的API地址
                data = '{"filepath":"/mnt/sda/COUNT/%s"}' % self.vid_url  # POST请求的数据，以字典形式提供
                requests.post(url, data=data, headers={'Content-Type': 'application/json'})
                # print(f"pid:{os.getpid()} 发送url: {url} data:{data}")
                logger.info(f"pid:{os.getpid()} 发送url: {url} data:{data}")
            except Exception as e:
                logger.error(f"pid:{os.getpid()} Not found valid ip or port: {self.webip}:{self.webport}")
                logger.error(e)


        except Exception as e:
            # 发生错误时回滚
            self.db.rollback()
            logger.error(f"pid:{os.getpid()} MySQL数据库错误: {e}")
        finally:
            cursor.close()


class KeliuSQLObj:
    def __init__(self, record_time, flow_cam_id, record_status, record_num, record_photo_url, is_warning,
                 record_video_url):
        # best_save_path = osp.join(*best_save_path.split('/')[1:]) # 去掉./YOLOX_outputs/
        self.table_name = "flow_record"
        if record_photo_url is not None and record_photo_url != "":
            record_photo_url = os.path.join(*record_photo_url.split('/')[1:])
        record_video_url = os.path.join(*record_video_url.split('/')[1:])
        self.sqlDict = {
            "record_time": record_time,
            "flow_cam_id": flow_cam_id,
            "record_status": record_status,
            "record_num": record_num,
            "record_photo_url": record_photo_url,
            "is_warning": is_warning,
            "record_video_url": record_video_url,
        }

    def get_dict(self):
        return self.sqlDict


class RenlianSQLObj:
    def __init__(self, record_time, recognize_cam_id, record_status, record_num, record_photo_url, personnel_id,
                 is_warning, record_video_url):
        # best_save_path = osp.join(*best_save_path.split('/')[1:]) # 去掉./YOLOX_outputs/
        self.table_name = "recognize_record"
        if record_photo_url is not None and record_photo_url != "":
            record_photo_url = os.path.join(*record_photo_url.split('/')[1:])
        record_video_url = os.path.join(*record_video_url.split('/')[1:])
        self.sqlDict = {
            "record_time": record_time,
            "recognize_cam_id": recognize_cam_id,
            "record_status": record_status,
            "record_num": record_num,
            "record_photo_url": record_photo_url,
            "personnel_id": personnel_id,
            "is_warning": is_warning,
            "record_video_url": record_video_url,
        }

    def get_dict(self):
        return self.sqlDict
