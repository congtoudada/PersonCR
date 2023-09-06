import enum
import shutil
import time
import os
import os.path as osp
from loguru import logger
import cv2
# from queue import Queue
import random
import queue

# 定义枚举类
Dir = enum.Enum('Dir', ('Null', 'In', 'Out'))


class MyCoutnerItem:
    # 默认往上离开（Out），往下进入（In）
    def __init__(self, init_dict):
        # 内部初始化
        self._end_ratio = None
        self._best_score = init_dict['conf']
        self._best_size = 0
        self._best_face_img = None
        # self._second_face_img = None  # 路径多个1 （两种来源：①进入检测区域；②有更好的图片）
        self._vid_writer = None
        self._dying = 0  # 0:运行 1:将亡 2:死亡
        self._det_flag = 0  # 里外包围盒模式下有效 0:先进入检测区 1:先进入缓冲区
        self._final_dir = None  # 最终计算得到的Dir
        self._is_success_img = False  # 是否成功抓拍
        self._trigger = 0  # 当前所处事件阶段 1:进入检测区 2:检测区中央 3:检测区外
        self._now_recorder = 0
        self._recorder_max = 15000 # 一个人最多录制 15000 帧 15000 = 25*60*10(10min+)
        self._cap_counter = 0 # 帧计数器，满足一定条件，会自动尝试抓拍
        self._cap_counter_max = init_dict['per_img_interval'] # 帧计数器上限
        self._last_vert_ratio = 0  # 当前对象上次自动抓拍时竖直占比
        self._last_hori_ratio = 0  # 当前对象上次自动抓拍时水平占比
        self._cap_move_thres = 0.1 # 移动阈值，竖直水平累计移动0.1才更新抓拍图
        # self._fail_cap_img = None # 没有抓拍到人脸时的截图

        # 外部初始化
        self.obj_id = init_dict['obj_id']  # 追踪id
        self.cam_id = init_dict['cam_id']  # 相机id
        self.run_mode = init_dict['run_mode']  # 运行模式 0:客流 1:人脸
        self.begin_ratio = init_dict['begin_ratio']  # 进入检测区域时所在位置
        self.begin_dir = init_dict['begin_dir']
        self.begin_time = init_dict['begin_time']  # 进入检测区域时间（单位：s）
        # time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.begin_time))
        self.begin_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(self.begin_time))  # 进入检测区域时间时间戳形式
        self.is_reverse = init_dict['is_reverse']
        # up_ratio = init_dict['up_ratio']
        # down_ratio = init_dict['down_ratio']
        # half_ratio = 0.5 * (up_ratio + down_ratio)
        self.save_path = osp.join(init_dict['save_path'], self.begin_timestamp)
        self.is_img = init_dict['is_img']
        self.is_vid = init_dict['is_vid']
        self.save_fps = init_dict['save_fps']  # 消亡态保留帧数
        self.cam_name = init_dict['cam_name']
        self.per_img_count = init_dict['per_img_count'] # 每个人保留的最大抓拍张数
        self.update_time = self.begin_time
        self.ltrb = (0, 0, 0, 0)  # 该帧对象的ltrb
        self.is_match = False  # 该帧是否参与匹配
        self.last_frame_id = 0  # 最后一次更新帧
        self.now_v_ratio = 0  # 当前y轴占比

        # 内部初始化:补充
        self._last_record_time = self.begin_time
        self._ready_face_queue = queue.Queue(self.per_img_count)  # 候选图队列


        os.makedirs(self.save_path, exist_ok=True)  # 创建目录

        # self.record_queue = Queue() # 抓拍视频缓存队列
        self.vid_save_path = ""
        if self.is_vid:
            self.cap_fps = init_dict['cap_fps']
            self.cap_width = int(init_dict['cap_width'])
            self.cap_height = int(init_dict['cap_height'])
            self.vid_save_path = osp.join(self.save_path, f"{self.obj_id}_{self.begin_dir.name}.mp4")
            # rint(f"cap_fps: {cap_fps} cap_width: {cap_width} cap_height: {cap_height}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # fourcc = cv2.VideoWriter_fourcc(*'H264')
            self._vid_writer = cv2.VideoWriter(
                self.vid_save_path,
                fourcc,
                self.cap_fps, (self.cap_width, self.cap_height))

    def set_det_flag(self):
        self._det_flag = 1

    def get_det_flag(self):
        return self._det_flag

    def get_trigger(self):
        return self._trigger

    def set_trigger(self, trigger):
        self._trigger = trigger

    def set_end_ratio(self, _end_ratio):
        self._end_ratio = _end_ratio

    def get_is_success_img(self):
        return self._is_success_img

    def set_best_img(self, face_img):
        self._is_success_img = True
        if self._best_face_img is not None:
            # self._second_face_img = self._best_face_img
            if self._ready_face_queue.full():
                self._ready_face_queue.get()
            self._ready_face_queue.put(self._best_face_img)  # 将旧的最佳候选图加入候选图队列中
        self._best_face_img = face_img

    def check_frame_counter(self, v_ratio, h_ratio):
        self._cap_counter += 1
        if self._cap_counter > self._cap_counter_max \
                and abs((v_ratio + h_ratio) - (self._last_vert_ratio + self._last_hori_ratio)) > self._cap_move_thres:
            self._cap_counter = 0
            self._last_vert_ratio = v_ratio
            self._last_hori_ratio = h_ratio
            return True
        else:
            return False

    # 会传入整个人的追踪图片
    # y_ratio=1.0 : 默认截全身
    def put_img(self, fail_img, y_ratio=1.0):
        if self._ready_face_queue.full():
            self._ready_face_queue.get()

        self._ready_face_queue.put(self._get_face_img(fail_img, (0, 0, int(fail_img.shape[1]), int(fail_img.shape[0])), border=0))
        # if y_ratio < 0.67:  # 画面靠上截上半身
        #     self._ready_face_queue.put(
        #         self._get_face_img(fail_img, (0, 0, int(fail_img.shape[1]), int(fail_img.shape[0] * 0.5)), border=0))
        # else:  # 画面靠下截全身
        #     self._ready_face_queue.put(
        #         self._get_face_img(fail_img, (0, 0, int(fail_img.shape[1]), int(fail_img.shape[0])), border=0))

    def check_best_score(self, score, img_size=None):
        if not self.is_img:
            return False
        if img_size is None:  # 忽略尺寸，单纯比较分数
            if score > self._best_score:
                # print(f"{self.obj_id} 置信度更新: {score} > {self._best_score}")
                self._best_score = score  # 更新置信度
                return True
        else:
            # 分数相差大，选分数大的
            if score - self._best_score > 0.008:
                self._best_score = score  # 更新置信度
                return True
            # 分数相近，选尺寸大的
            elif self._best_score - score < 0.008:
                if img_size > self._best_size:
                    self._best_size = img_size
                    return True

        return False

    def calFace(self, ltrb, score, online_im, border=5):
        if self.is_match: # 该帧参与匹配，则不继续匹配
            return
        # print(f"人脸检测 脸:{ltrb} 人:{self.ltrb}")
        if self.ltrb[0] < ltrb[0] and self.ltrb[1] < ltrb[1] and self.ltrb[2] > ltrb[2] and self.ltrb[3] > ltrb[3]:
            # print(f"人脸检测成功: {ltrb}")
            img_size = min(ltrb[2] - ltrb[0], ltrb[3] - ltrb[1])
            self.is_match = True
            if self.check_best_score(score, img_size):
                self.set_best_img(self._get_face_img(online_im, ltrb, border))

    def _get_face_img(self, im, ltrb, border=1):
        x1, y1, x2, y2 = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
        x1 = x1 - border
        y1 = y1 - border
        x2 = x2 + border
        y2 = y2 + border
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = im.shape[1] if x2 > im.shape[1] else x2
        y2 = im.shape[0] if y2 > im.shape[0] else y2
        return im[int(y1): int(y2), int(x1): int(x2)]

    def set_dying(self, _dying):
        # print(f"对象状态变更: {_dying}")
        self._dying = _dying

    def set_update_time(self, now):
        self.update_time = now

    def get_update_time(self):
        return self.update_time

    def get_dying(self):
        return self._dying

    def click_save_fps(self):
        self.save_fps -= 1
        if self.save_fps <= 0:
            self.set_dying(2)

    # 视频录制
    def record_video(self, im, now):
        if self.is_vid:
            self._now_recorder += 1
            if self._now_recorder > self._recorder_max:
                return
            # if now - self._last_record_time < 3:  # 两次写间隔小于3s才记录
            # print(f"before resize img shape: {im.shape}")
            re_img = cv2.resize(im, (self.cap_width, self.cap_height))
            # print(f"after resize img shape: {re_img.shape}")
            self._vid_writer.write(re_img)
            # self._last_record_time = now

    # 有效离开时调用（保存图片/视频，写数据库）
    def on_success_quit(self, now, is_sql=False, qface_list=None, qsql_list=None, fail_img=None):
        if self.save_path == "":
            return

        # quit_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        # 保存最高分图片
        best_save_path = ""
        if self.is_img:
            best_save_path = osp.join(self.save_path, f"{self.obj_id}_{self.get_dir().name}.jpg")
            if self._is_success_img:
                # best_save_path = osp.join(self.save_path, f"{self.obj_id}_{self.get_dir().name}.png")
                # print(f"img: {self._best_face_img}")
                cv2.imwrite(best_save_path, self._best_face_img)
            else:
                cv2.imwrite(best_save_path, self._get_face_img(fail_img, (0, 0, int(fail_img.shape[1]), int(fail_img.shape[0]))))
                print(f"{self.cam_name} {self.obj_id} 没有抓拍到图片！")
                logger.info(f"{self.cam_name} {self.obj_id} 没有抓拍到图片！")
            # 不论成功与否，都有候选图
            qsize = self._ready_face_queue.qsize()
            for i in range(self._ready_face_queue.qsize()):
                second_save_path = osp.join(self.save_path, f"{self.obj_id}_{self.get_dir().name}{qsize-i}.jpg")  # 多个1
                cv2.imwrite(second_save_path, self._ready_face_queue.get()) # 从最差候选图开始写

        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
        personnel_id = 1

        # print(f"run_mode: {self.run_mode}")
        if self.run_mode == 0 and is_sql:
            # 数据库进程调度：随机调度
            index = random.randrange(0, len(qsql_list))  # 包含下限，不包含上限（下闭上开）
            if qsql_list[index].full():  # 如果队列满了，就尝试更换一下
                index = (index + 1) % len(qsql_list)
            p_qsql = qsql_list[index]

            if not p_qsql.full():
                # 直接传递给数据库进程
                p_qsql.put({"run_mode": self.run_mode,
                            # 数据库字段
                            "record_time": time_str,
                            "flow_cam_id": self.cam_id,
                            "record_status": 0 if self._final_dir == Dir.In else 1,
                            "record_num": 1,
                            "record_photo_url": best_save_path,
                            "is_warning": 0,
                            "record_video_url": self.vid_save_path,
                            })
            else:
                logger.error("Mysql Queue is full! Maybe MySQL is closed!")
        elif self.run_mode == 1:
            # 无论是否抓拍到人脸，都送过去人脸识别
            # if self._is_success_img:
            # 人脸识别进程调度：空闲优先，随机，满了则换
            p_qface = None
            for i in range(len(qface_list)):
                if qface_list[i].empty():
                    p_qface = qface_list[i]
                    break

            if p_qface is None:
                index = random.randrange(0, len(qface_list))  # 包含下限，不包含上限（下闭上开）
                index_flag = index
                while qface_list[index].full():  # 如果队列满了，就尝试更换一下
                    index = (index + 1) % len(qface_list)
                    if index_flag == index:
                        print("Fatal Error!All Face Queue is full! Maybe Face is closed or too busing!")
                        logger.error("Fatal Error!All Face Queue is full! Maybe Face is closed or too busing!")
                        break
                p_qface = qface_list[index]

            if not p_qface.full():
                # print(f"cam_id: {self.cam_id}")
                # 如果存在抓拍图片，就先传递给人脸识别进程，再由人脸识别进程传递给数据库进程
                p_qface.put({"run_mode": self.run_mode,
                             "is_sql": is_sql,
                             "obj_id": self.obj_id,
                             # 数据库字段
                             "record_time": time_str,
                             "recognize_cam_id": self.cam_id,
                             "record_status": 0 if self._final_dir == Dir.In else 1,
                             "record_num": 1,
                             "record_photo_url": best_save_path,
                             "personnel_id": personnel_id,
                             "is_warning": 0,
                             "record_video_url": self.vid_save_path,
                             })

    def set_dir(self, dir):
        self._final_dir = dir

    def get_dir(self):
        if self._final_dir is not None:
            return self._final_dir

        if self.begin_dir is None:
            return Dir.Null

        # dir = Dir.Null  # 利用begin_ratio和_end_ratio计算的Dir
        if self._end_ratio < self.begin_ratio:
            dir = Dir.Out
            if self.is_reverse:
                dir = Dir.In
        else:
            dir = Dir.In
            if self.is_reverse:
                dir = Dir.Out

        # print(f"dir: {dir} - begin_dir: {self.begin_dir}")
        if dir == self.begin_dir:
            self._final_dir = dir
            return dir
        else:
            self._final_dir = Dir.Null
            return Dir.Null

    def on_destroy(self):
        if self._vid_writer is not None:
            self._vid_writer.release()

    # 删除录像资源和文件夹
    def clear_resource(self):
        if self.is_vid:
            self._vid_writer.release()

        # 删除该对象所属录像资源
        # shutil.rmtree(self.save_path)
        if self.is_vid:
            if os.path.exists(self.vid_save_path):
                os.remove(self.vid_save_path)
        if len(os.listdir(self.save_path)) == 0:  # 目录为空，则删除
            os.rmdir(self.save_path)

        logger.info(f"删除数据: {self.obj_id}")
