# import os
#
# import cv2
#
# from tools.lzc.count.framework.CountEnum import FaceStateEnum, CountItemStateEnum, ZoneEnum
# from tools.lzc.count.framework.CountItem import CountItem
# from tools.lzc.count.framework.CountMgrData import CountMgrRunningData, CountMgrData
# from loguru import logger
#
# from tools.lzc.face.framework.FaceRegTool import FaceRegTool
# from tools.lzc.sql.framework.SqlTool import SqlTool
#
#
# class FaceRegItem(CountItem):
#     def __init__(self):
#         super().__init__()
#         self.per_id = 0
#         self.current_per_id = 0
#         self.score = 0
#         self.reg_results = {}
#         self.last_face_req_frame = 0
#         self.is_req = False
#         # 人脸模式成功离开延迟发送请求
#         self.face_state = FaceStateEnum.NoSend
#
#     def init(self, countMgr, running_data: CountMgrRunningData, begin_zone, begin_state):
#         super().init(countMgr, running_data, begin_zone, begin_state)
#         self.per_id = 1  # 默认为陌生人
#         self.current_per_id = 1  # 当前识别情况
#         self.score = 0  # 识别分数
#         self.reg_results.clear()
#         self.last_face_req_frame = 0
#         self.is_req = False  # 是否在发送人脸请求
#         # 人脸模式成功离开延迟发送请求
#         self.face_state = FaceStateEnum.NoSend
#
#     def detect_update(self, data: CountMgrData, running_data: CountMgrRunningData):
#         super().detect_update(data, running_data)
#
#     def buffer_update(self, data: CountMgrData, running_data: CountMgrRunningData):
#         # super().buffer_update(data, running_data)
#         # 发送人脸识别请求
#         if self.state == CountItemStateEnum.Valid:
#             # 当前为陌生人且超过发送间隔才发送
#             if self.per_id == 1 and not self.is_req and running_data.frame_id - self.last_face_req_frame > data.reg_frequency:
#                 self.last_face_req_frame = running_data.frame_id
#                 if self._send_reg_req(self.obj_id, running_data.im, running_data.tlwh):
#                     self.is_req = True
#
#     def global_update(self, im, now, frame_id):
#         super().global_update(im, now, frame_id)
#         if self.state == CountItemStateEnum.Dying:
#             # 延迟发送人脸消息(如果有请求，则收到响应再发送)
#             if not self.is_req and self.face_state == FaceStateEnum.CanSend:
#                 self._on_renlian_succe_quit()
#
#     # 人脸识别成功事件，会根据情况将图片写入本地
#     def process_reg_rsp(self, per_id, score, face_img):
#         self.current_per_id = per_id
#         self.is_req = False
#         if per_id == 1:
#             return
#
#         if not self.reg_results.__contains__(per_id):
#             self.reg_results[per_id] = 1
#         else:
#             self.reg_results[per_id] += 1
#
#         data: CountMgrData = self.countMgr.data
#         # 匹配成功超过一定次数，才认为匹配成功
#         if self.reg_results[per_id] >= data.reg_count_thresh:
#             self.per_id = per_id
#             self.score = score
#             logger.info(f"{self.countMgr.data.cam_name} 成功找到人脸匹配结果: {self.obj_id} <=> {self.per_id}")
#             # 写入本地
#             if data.is_img:
#                 self.img_save_path = os.path.join(self.save_dir,
#                                                   f"{self.obj_id}_{self.begin_zone.name}_{per_id}_{score:.2f}.jpg")
#                 cv2.imwrite(self.img_save_path, face_img)
#
#     # 成功离开事件
#     def on_success_quit(self, data: CountMgrData, running_data: CountMgrRunningData):
#         # super().on_success_quit(data, running_data)
#         self.state = CountItemStateEnum.Dying
#         self.face_state = FaceStateEnum.CanSend
#
#         # 生成兜底图
#         if self.per_id == 1:
#             super()._create_shot_img(data, running_data)
#
#     # 人脸识别请求
#     def _send_reg_req(self, obj_id, im, tlwh):
#         data: CountMgrData = self.countMgr.data
#         face_img = self._get_track_img(im, tlwh, data.border)
#         pack_data = FaceRegTool.pack_req(obj_id, face_img)
#
#         if data.qface_req and not data.qface_req.full():
#             data.qface_req.put(pack_data)
#             return True
#         else:
#             return False
#
#     # 发送人脸数据给数据库
#     def _on_renlian_succe_quit(self):
#         if self.face_state == FaceStateEnum.CanSend:
#             self.face_state = FaceStateEnum.Sended
#             data: CountMgrData = self.countMgr.data
#
#             # 人脸识别优化，增大匹配几率
#             # 如果self.per_id==1，则在符合阈值的结果中挑选最出现次数最多的
#             if self.per_id == 1:
#                 # 取出字典中值最大的元素对应的键
#                 if self.reg_results.__len__() > 0:
#                     self.per_id = max(self.reg_results, key=self.reg_results.get)
#
#             pack_data = SqlTool.pack_renlian_data(
#                 record_time=self.begin_time,
#                 recognize_cam_id=data.cam_id,
#                 record_status=0 if self.begin_zone == ZoneEnum.In else 1,
#                 record_num=1,
#                 record_photo_url=self.img_save_path,
#                 personnel_id=self.per_id,
#                 is_warning=0,
#                 record_video_url=self.vid_save_path
#             )
#             self._send_sql_req(data, pack_data)
#
#     # 清除不合法状态下的资源
#     def on_release(self, is_clear):
#         self._on_renlian_succe_quit()  # 延迟发送人脸消息
#         super().on_release(is_clear)
