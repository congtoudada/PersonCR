import os
import time

import cv2

from tools.lzc.count.framework.CountEnum import ZoneEnum, CountItemStateEnum, FaceStateEnum
from tools.lzc.count.framework.CountMgrData import CountMgrData, CountMgrRunningData
from tools.lzc.face.framework.FaceRegTool import FaceRegTool
from tools.lzc.sql.framework.SqlTool import SqlTool
from loguru import logger


class CountItem:
    def __del__(self):
        if self.vid_writer:
            self.vid_writer.release()

    def __init__(self):
        self.countMgr = None
        self.obj_id = 0
        self.begin_point = (0, 0)
        self.begin_time = 0
        self.begin_frame_id = 0
        self.begin_zone = ZoneEnum.Null
        self.state = CountItemStateEnum.Dead

        # 内部初始化
        self.vid_writer = None
        self.has_recorder = 0  # 已经记录的帧数
        self.save_dir = ""  # 保存目录
        self.dying_has_recorder = 0  # Dying状态下已经记录的帧数
        self.img_save_path = ""  # img_save_path 在 on_success_reg内设置
        self.vid_save_path = ""
        self.last_update_time = 0
        self.last_update_frame = 0
        # 人脸 extension
        self.per_id = 0
        self.current_per_id = 0
        self.score = 0
        self.best_face_img = None
        self.reg_results = {}
        self.last_face_req_frame = 0
        self.req_count = 0
        # 人脸模式成功离开延迟发送请求
        self.face_state = FaceStateEnum.NoSend

    def init(self, countMgr, running_data: CountMgrRunningData, begin_zone, begin_state):
        self.countMgr = countMgr
        self.obj_id = running_data.obj_id
        self.begin_point = running_data.point
        self.begin_time = running_data.now
        self.begin_frame_id = running_data.frame_id
        self.begin_zone = begin_zone
        self.state = begin_state

        # 内部初始化
        data: CountMgrData = countMgr.data
        if self.vid_writer:
            self.vid_writer.release()
            self.vid_writer = None
        begin_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(self.begin_time))  # 进入检测区域时间时间戳形式
        self.save_dir = os.path.join(data.save_path, f"{begin_timestamp}_{self.obj_id}")
        self.has_recorder = 0
        self.dying_has_recorder = 0
        self.img_save_path = ""
        self.vid_save_path = ""
        self.last_update_time = running_data.now
        self.last_update_frame = running_data.frame_id
        # 人脸extension
        self.per_id = 1  # 默认为陌生人
        self.current_per_id = 1  # 当前识别情况 (用于可视化)
        self.score = 0  # 识别分数
        self.best_face_img = None  # 最好的识别截图
        self.reg_results.clear()
        self.last_face_req_frame = running_data.frame_id
        self.req_count = 0  # 当前发送的face req数量，每收到一次响应会-1
        # 人脸模式成功离开延迟发送请求
        self.face_state = FaceStateEnum.NoSend

        # 合法状态，创建资源保存目录
        if self.state == CountItemStateEnum.Valid:
            os.makedirs(self.save_dir, exist_ok=True)
            if data.is_vid:
                self.vid_save_path = os.path.join(self.save_dir, f"{self.obj_id}_{self.begin_zone.name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # fourcc = cv2.VideoWriter_fourcc(*'H264')
                self.vid_writer = cv2.VideoWriter(self.vid_save_path, fourcc, data.cap_fps,
                                                  (data.cap_width, data.cap_height))

    def detect_update(self, data: CountMgrData, running_data: CountMgrRunningData):
        pass

    def buffer_update_valid(self, data: CountMgrData, running_data: CountMgrRunningData):
        # ping
        self.last_update_time = running_data.now
        self.last_update_frame = running_data.frame_id

        # 录制视频
        self._record_video(running_data.history_im, running_data.now, running_data.frame_id)

        # 发送人脸识别请求
        if data.run_mode == 1:
            # 当前分数小于期望则继续发送 没有正在发送的请求 and 超过发送间隔才发送
            if self.req_count <= 0 \
                    and running_data.frame_id - self.last_face_req_frame > data.reg_interval \
                    and self.score < data.reg_score_expect:
                if self._send_reg_req(self.obj_id, running_data.im, running_data.tlwh):
                    self.last_face_req_frame = running_data.frame_id
                    self.req_count += 1

    def buffer_update_invalid(self, data: CountMgrData, running_data: CountMgrRunningData):
        # ping
        self.last_update_time = running_data.now
        self.last_update_frame = running_data.frame_id

    def global_update(self, im, now, frame_id):
        if self.state == CountItemStateEnum.Dying:
            self.last_update_time = now
            self.last_update_frame = frame_id
            # 录制视频
            self._record_video(self.countMgr.frame_history.queue[0], now, frame_id)
            # 延迟发送人脸消息(如果有请求，则收到响应再发送)
            if self.countMgr.data.run_mode == 1 \
                    and self.face_state == FaceStateEnum.CanSend \
                    and self.req_count <= 0:
                self._on_renlian_succe_quit()

    # 人脸识别请求
    def _send_reg_req(self, obj_id, im, tlwh):
        data: CountMgrData = self.countMgr.data
        face_img = self._get_track_img(im, tlwh, data.border)
        pack_data = FaceRegTool.pack_req(obj_id, face_img)

        if data.qface_req and not data.qface_req.full():
            data.qface_req.put(pack_data)
            # print(f"send face req: {self.obj_id} {self.countMgr.running_data.frame_id}")
            # cv2.imwrite(os.path.join(self.save_dir, f"{self.countMgr.running_data.frame_id}.jpg"), face_img)
            return True
        else:
            return False

    # 人脸识别成功事件，会根据情况将图片写入本地
    def process_reg_rsp(self, per_id, score, face_img):
        self.current_per_id = per_id
        self.req_count -= 1
        # print(f"receive face rsp: {self.obj_id} {self.countMgr.running_data.frame_id}")
        if per_id == 1:  # 陌生人则返回
            return

        if not self.reg_results.__contains__(per_id):
            self.reg_results[per_id] = 1
        else:
            self.reg_results[per_id] += 1

        data: CountMgrData = self.countMgr.data
        # 匹配成功超过一定次数，才认为匹配成功
        if self.reg_results[per_id] >= data.reg_count_thresh:
            self.per_id = per_id
            # 分数更新
            if self.score < score:
                self.score = score
                self.best_face_img = face_img

            logger.info(f"{self.countMgr.data.cam_name} 成功找到人脸匹配结果: {self.obj_id} <=> {self.per_id}")
            # # 写入本地
            # if data.is_img:
            #     self.img_save_path = os.path.join(self.save_dir,
            #                                       f"{self.obj_id}_{self.begin_zone.name}_{per_id}_{score:.2f}.jpg")
            #     cv2.imwrite(self.img_save_path, face_img)

    # 视频录制
    def _record_video(self, im, now, frame_id):
        data: CountMgrData = self.countMgr.data
        # 处于将亡态录制视频，达到一定值会进入死亡态
        if self.state == CountItemStateEnum.Dying:
            self.dying_has_recorder += 1
            if self.dying_has_recorder >= data.save_frames:
                self.state = CountItemStateEnum.Dead
        # 如果可以录制视频则录制
        if data.is_vid:
            if self.has_recorder <= data.max_record_frames:
                re_img = cv2.resize(im, (data.cap_width, data.cap_height))
                self.vid_writer.write(re_img)
                # cv2.imwrite(os.path.join(self.save_dir, f"{frame_id}.jpg"), re_img)
                self.has_recorder += 1

    # 成功离开事件
    def on_success_quit(self, data: CountMgrData, running_data: CountMgrRunningData):
        # 设为Dying状态
        self.state = CountItemStateEnum.Dying
        # 生成截图
        self._create_shot_img(data, running_data)

        if data.run_mode == 0:
            self._on_keliu_success_quit(data, running_data)  # 直接发送消息给数据库
        else:
            if self.per_id == 1:  # 如果是陌生人，则继续尝试识别
                self.face_state = FaceStateEnum.CanSend  # 标记为可发消息的状态
                # 立即发送一次人脸识别请求
                if self._send_reg_req(self.obj_id, running_data.im, running_data.tlwh):
                    self.last_face_req_frame = running_data.frame_id
                    self.req_count += 1
            else:  # 如果不是陌生人，则立即发送给数据库
                self.face_state = FaceStateEnum.CanSend
                self._on_renlian_succe_quit()

    def _on_keliu_success_quit(self, data: CountMgrData, running_data: CountMgrRunningData):
        pack_data = SqlTool.pack_keliu_data(
            record_time=self.begin_time,
            flow_cam_id=data.cam_id,
            record_status=0 if self.begin_zone == ZoneEnum.In else 1,
            record_num=1,
            record_photo_url=self.img_save_path,
            is_warning=0,
            record_video_url=self.vid_save_path
        )
        logger.info(f"{data.cam_name} {self.obj_id}:发送keliu数据到数据库 {self.begin_zone}")
        self._send_sql_req(data, pack_data)

    def _on_renlian_succe_quit(self):
        if self.face_state == FaceStateEnum.CanSend:
            self.face_state = FaceStateEnum.Sended
            data: CountMgrData = self.countMgr.data

            # # 人脸识别优化，增大匹配几率（视情况开启）：
            # #   如果self.per_id==1，则在符合阈值的结果中挑选最出现次数最多的
            # if self.per_id == 1:
            #     # 取出字典中值最大的元素对应的键
            #     if self.reg_results.__len__() > 0:
            #         # 使用max函数和字典的values方法找到最大的值，然后直接使用该最大值在字典中查找对应的键
            #         max_value = max(self.reg_results.values())
            #         self.per_id = next(key for key, value in self.reg_results.items() if value == max_value)

            pack_data = SqlTool.pack_renlian_data(
                record_time=self.begin_time,
                recognize_cam_id=data.cam_id,
                record_status=0 if self.begin_zone == ZoneEnum.In else 1,
                record_num=1,
                record_photo_url=self.img_save_path,
                personnel_id=self.per_id,
                is_warning=0,
                record_video_url=self.vid_save_path
            )
            logger.info(f"{data.cam_name} {self.obj_id}:发送renlian数据到数据库 per_id={self.per_id} {self.begin_zone}")
            self._send_sql_req(data, pack_data)

    def _send_sql_req(self, data: CountMgrData, pack_data: dict):
        # 传递给数据库进程
        if data.qsql_list:
            lst = [qsql.qsize() for qsql in data.qsql_list]
            # 找出最小元素
            min_element = min(lst)
            # 找出最小元素的下标
            index_of_min_element = lst.index(min_element)
            data.qsql_list[min_element].put(pack_data)

    # 成功离开预处理
    def _create_shot_img(self, data: CountMgrData, running_data: CountMgrRunningData):
        # 生成兜底图
        # logger.info(f"{data.cam_name} 生成候选图: {self.obj_id}_{self.begin_zone.name}.jpg")
        if data.run_mode == 0:
            self.img_save_path = os.path.join(self.save_dir, f"{self.obj_id}_{self.begin_zone.name}.jpg")
            cv2.imwrite(self.img_save_path, self._get_track_img(running_data.im, running_data.tlwh, data.border))
        else:
            self.img_save_path = os.path.join(
                self.save_dir, f"{self.obj_id}_{self.begin_zone.name}_{self.per_id}_{self.score:.2f}.jpg.jpg")
            if self.per_id == 1:
                cv2.imwrite(self.img_save_path, self._get_track_img(running_data.im, running_data.tlwh, data.border))
            else:
                cv2.imwrite(self.img_save_path, self.best_face_img)


    # 清除不合法状态下的资源
    def on_release(self, is_clear):
        self.state = CountItemStateEnum.Dead

        # 如果对象离开buffer不久重新返回，需要立即写数据库
        if self.countMgr.data.run_mode == 1:
            self._on_renlian_succe_quit()

        if self.vid_writer:
            self.vid_writer.release()
            self.vid_writer = None

        if is_clear:
            # 删除该对象所属录像资源
            if os.path.exists(self.vid_save_path):
                os.remove(self.vid_save_path)
            if os.path.exists(self.save_dir) and len(os.listdir(self.save_dir)) == 0:  # 目录为空，则删除
                os.rmdir(self.save_dir)

    def _get_track_img(self, im, tlwh, border=1):
        x1, y1, w, h = tlwh
        x2 = x1 + w + border
        x1 = x1 - border
        y2 = y1 + h + border
        y1 = y1 - border
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = im.shape[1] if x2 > im.shape[1] else x2
        y2 = im.shape[0] if y2 > im.shape[0] else y2
        return im[int(y1): int(y2), int(x1): int(x2)]
