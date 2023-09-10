# SQL和Web后端交互
import os
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

    async def _send_request(self, url: str):
        async with aiohttp.ClientSession() as session:
            await session.get(url)

    async def _send_post_request(self, url, data, headers):
        async with aiohttp.ClientSession() as session:
            await session.post(url, data=data, headers=headers)

    async def send_renlian_alarm(self, last_id, web_ip=None, web_port=None):
        webip, webport = self._check_ip(web_ip, web_port)
        try:
            url = f"http://{webip}:{webport}/recognizeRecordEntity/setAlramByRecordId?recordId={last_id}"
            logger.info(f"pid:{os.getpid()} request: {url}")
            await self._send_request(url)
        except Exception as e:
            logger.error(f"pid:{os.getpid()} request错误: {e}")

    async def send_keliu_alarm(self, last_id, web_ip=None, web_port=None):
        webip, webport = self._check_ip(web_ip, web_port)
        try:
            url = f"http://{webip}:{webport}/flowRecordEntity/setAlramByRecordId?recordId={last_id}"
            logger.info(f"pid:{os.getpid()} request: {url} id:{last_id}")
            await self._send_request(url)
        except Exception as e:
            logger.error(f"pid:{os.getpid()} request错误: {e}")

    async def send_change_format(self, vid_url, web_ip=None):
        webip, webport = self._check_ip(web_ip, None)
        try:
            # 发送后端，要求视频转码
            url = f"http://{webip}:9092/convert_video"  # 替换为实际的API地址
            data = '{"filepath":"/mnt/sda/COUNT/%s"}' % vid_url  # POST请求的数据，以字典形式提供
            logger.info(f"pid:{os.getpid()} post: {url} data:{data}")
            await self._send_post_request(url, data=data, headers={'Content-Type': 'application/json'})
        except Exception as e:
            logger.error(f"pid:{os.getpid()} post错误: {e}")



if __name__ == "__main__":
    log_path = os.path.join("logs", "{time:YYYY-MM-DD}_main_test" +".log")
    logger.add(sink=log_path, rotation="daily")  # 每100MB重新写

    # 没有异步发生，无能为力
    logger.info("begin")
    async def mytest():
        await asyncio.sleep(3)
        logger.info("async operation completed")
    async def mytest2():
        await mytest()
    asyncio.run(mytest2())
    logger.info("over")
    time.sleep(3)