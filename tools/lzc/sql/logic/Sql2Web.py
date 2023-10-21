# SQL和Web后端交互
import os
import threading
import time

import requests
import urllib.request

import aiohttp
import asyncio


from loguru import logger


class Sql2Web():
    def __init__(self, web_ip, web_port):
        self.web_ip = web_ip
        self.web_port = web_port

    def _check_ip(self, web_ip, web_port):
        webip = self.web_ip
        webport = self.web_port
        if web_ip is not None:
            webip = web_ip
        if web_port is not None:
            webport = web_port
        return webip, webport

    def send_renlian_alarm(self, last_id, web_ip=None, web_port=None):
        webip, webport = self._check_ip(web_ip, web_port)
        try:
            url = f"http://{webip}:{webport}/recognizeRecordEntity/setAlramByRecordId?recordId={last_id}"
            urllib.request.urlopen(url)
            logger.info(f"pid:{os.getpid()} request: {url}")
        except Exception as e:
            logger.error(f"pid:{os.getpid()} request错误: {e}")

    def send_keliu_alarm(self, last_id, web_ip=None, web_port=None):
        webip, webport = self._check_ip(web_ip, web_port)
        try:
            url = f"http://{webip}:{webport}/flowRecordEntity/setAlramByRecordId?recordId={last_id}"
            urllib.request.urlopen(url)
            logger.info(f"pid:{os.getpid()} request: {url} id:{last_id}")
        except Exception as e:
            logger.error(f"pid:{os.getpid()} request错误: {e}")

    def send_change_format(self, vid_url, web_ip=None):
        webip, webport = self._check_ip(web_ip, None)
        try:
            # 发送后端，要求视频转码
            url = f"http://{webip}:9092/convert_video"  # 替换为实际的API地址
            data = '{"filepath":"/mnt/sda/COUNT/%s"}' % vid_url  # POST请求的数据，以字典形式提供
            requests.post(url, data=data, headers={'Content-Type': 'application/json'})
            logger.info(f"pid:{os.getpid()} post: {url} data:{data}")
        except Exception as e:
            logger.error(f"pid:{os.getpid()} post错误: {e}")




if __name__ == "__main__":
    log_path = os.path.join("logs", "{time:YYYY-MM-DD}_main_test" +".log")
    logger.add(sink=log_path, rotation="daily")  # 每100MB重新写

    s2w = Sql2Web("127.0.0.1", 80)

    logger.info("begin")

    def my_function(a):
        s2w.send_keliu_alarm(1)

    thread2 = threading.Thread(target=my_function, args=(1,))
    thread2.start()

    # async def mytest():
    #     await asyncio.sleep(3)
    #     logger.info("async operation completed")
    # async def mytest2():
    #     await mytest()
    # asyncio.run(mytest2())
    logger.info("over")
    time.sleep(3)