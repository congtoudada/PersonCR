import numpy as np

from loguru import logger
from .counter_item import *


class MyCounterMgr:
    # 当检测对象进入检测区域时才视为进入，离开缓冲区域时才视为离开
    def __init__(self, init_dict):
        # 内部初始化
        self._in_count = 0  # 进人数
        self._out_count = 0  # 出人数
        self._destroy_time = 10  # 如果对象消失5s以上，就销毁
        # self._black_time = 60 * 60 * 6  # 如果存在X s没有销毁，就强制销毁
        self._last_schedule_sec1 = 0  # 秒计时器标志
        self._last_schedule_sec2 = 0  # 秒计时器标志
        self._last_schedule_hour1 = 0  # 小时计时器标志
        self._last_schedule_frame = 0  # 帧计时器标志
        # self.frame_id = 0  # 当前帧数
        self.lost_frame = init_dict['lost_frame']  # 对消消失多少帧销毁
        # self._face_flag = 0  # 检测人脸flag
        # self._face_interval = 1  # 每x帧检测一次人脸
        self.counter_dict = {}  # counter_item的容器

        # 外部初始化
        # 来自配置文件
        self.run_mode = init_dict['run_mode']  # 0：客流 1：人脸
        self.cam_name = init_dict['cam_name']  # 相机名称
        self.cal_mode = init_dict['cal_mode']  # 算法模式 0:左上角为基准点 1:中心为基准点
        self.det_mode = init_dict['det_mode']  # 检测模式 0:常规 1:里外包围盒
        self.cam_id = init_dict['cam_id']  # 相机id
        self.center_ratio = init_dict['center_ratio']  # 中线位置比例 (0:up_ratio 1:down_ration)
        self.up_ratio = init_dict['up_ratio']  # 上部：检测区域
        self.down_ratio = init_dict['down_ratio']  # 下部：检测区域
        self.half_ratio = self.up_ratio * (1 - self.center_ratio) + self.down_ratio * self.center_ratio
        self.buffer_up_ratio = init_dict['buffer_up_ratio']  # 上部：缓冲区域
        self.buffer_down_ratio = init_dict['buffer_down_ratio']  # 下部：缓冲区域
        self.left_ratio = init_dict['left_ratio']  # 左部：检测区域
        self.right_ratio = init_dict['right_ratio']  # 右部：检测区域
        self.buffer_left_ratio = init_dict['buffer_left_ratio']  # 左部：缓冲区域
        self.buffer_right_ratio = init_dict['buffer_right_ratio']  # 右部：缓冲区域
        self.is_reverse = init_dict['is_reverse']  # 是否反向 [以竖直方向为例：默认:往上离开(Out) 反向:往下离开(Out)]
        self.is_vertical = init_dict['is_vertical']  # 是否竖直检测
        self.only_dir = init_dict['only_dir']  # 是否只检测一个方向 0:双向检测 1:只检测进 2:只检测出
        self.lzc_debug = init_dict['lzc_debug']  # 是否debug
        # self.face_interval = init_dict['face_interval']  # 进入检测区域，每x帧执行一次人脸检测
        self.is_sql = init_dict['is_sql']  # 是否写数据库
        self.border = init_dict['border']  # 抓拍边框大小，该值越大抓拍尺寸越大
        # 抓拍参数
        self.is_img = init_dict['is_img']  # 是否抓拍图像
        self.is_vid = init_dict['is_vid']  # 是否抓拍视频
        self.cap_fps = init_dict['cap_fps']  # 抓拍帧率
        self.cap_width = init_dict['cap_width']  # 抓拍宽
        self.cap_height = init_dict['cap_height']  # 抓拍高
        self.save_fps = init_dict['save_fps']  # 抓拍视频时，保留的额外帧数
        self.is_vis = init_dict['is_vis']
        self.conf = init_dict['conf']
        self.per_img_count = init_dict['per_img_count']  # 每个人最大抓拍图张数
        self.per_img_interval = init_dict['per_img_interval']  # 如果没有抓拍到合适人脸，就隔x帧自动抓拍

        # 来自源码
        self.save_path = init_dict['save_path']

        # 其他
        self.qface_list = init_dict['qface_list']  # 进程队列：人脸
        self.qsql_list = init_dict['qsql_list']  # 进程队列：数据库
        self.face_predictor = init_dict['face_predictor']  # 人脸检测器
        self.test_size = init_dict['test_size']  # 人脸输入
        # self.can_detect_face = False

    # 绘制计数结果
    def draw_count(self, im):
        if self.lzc_debug:
            cv2.putText(im, "In: %d - Out: %d" % (self._in_count, self._out_count),
                        (8, im.shape[0] - int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                        thickness=2)

    def update(self, im, tlwhs, obj_ids, scores, now, frame_id):
        im = np.ascontiguousarray(np.copy(im))
        im_h, im_w = im.shape[:2]

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            obj_id = int(obj_ids[i])
            # 追踪对象状态更新
            if self.cal_mode == 0:  # 左上角检测
                self._update(obj_id, float(y1 / im_h), float(x1 / im_w), scores[i], im, tlwh,
                             now=now, frame_id=frame_id)
                if self.is_vis:
                    cv2.circle(im, (int(x1), int(y1)), 4, (118, 154, 242), 2)
            else:  # 中心检测
                self._update(obj_id, float((y1 + 0.5 * h) / im_h), float((x1 + 0.5 * w) / im_w), scores[i], im,
                             tlwh, now=now, frame_id=frame_id)
                if self.is_vis:
                    cv2.circle(im, (int(x1 + 0.5 * w), int(y1 + 0.5 * h)), 4, (118, 154, 242), 2)
        return im

    def draw_zone(self, im=None):
        if not self.is_vis:
            return im
        if im is None:
            print("im is None, Can't draw Zone!")
            return im
        im_h, im_w = im.shape[:2]
        text_scale = 2
        text_thickness = 2
        line_thickness = 2
        if self.is_vertical:  # 是否竖直检测
            # 绘制检测区域
            zone_color = (118, 154, 242)  # 朱颜酡 cv2: BGR (118, 154, 242)
            cv2.rectangle(im, pt1=(int(self.left_ratio * im_w), int(self.up_ratio * im_h)),
                          pt2=(int(self.right_ratio * im_w), int(self.down_ratio * im_h)), color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Detection", (0, int(self.up_ratio * im_h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        zone_color,
                        thickness=text_thickness)
            # 绘制检测区域-中线
            if self.det_mode == 0:
                zone_color = (130, 107, 220)  # 长春 BGR
                startPoint = (int(self.left_ratio * im_w), int(self.half_ratio * im_h))
                endPoint = (int(self.right_ratio * im_w), int(self.half_ratio * im_h))
                cv2.line(im, startPoint, endPoint, zone_color, thickness=2)

            # 绘制缓冲区域
            zone_color = (182, 182, 178)  # 月魄 BGR (182, 182, 178)
            cv2.rectangle(im, pt1=(int(self.buffer_left_ratio * im_w), int(self.buffer_up_ratio * im_h)),
                          pt2=(int(self.buffer_right_ratio * im_w), int(self.buffer_down_ratio * im_h)),
                          color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Buffer", (0, int(self.buffer_up_ratio * im_h)), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, zone_color,
                        thickness=text_thickness)
        else:
            # 绘制检测区域
            zone_color = (118, 154, 242)  # 朱颜酡 cv2: BGR (118, 154, 242)
            cv2.rectangle(im, pt1=(int(self.up_ratio * im_w), int(self.left_ratio * im_h)),
                          pt2=(int(self.down_ratio * im_w), int(self.right_ratio * im_h)), color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Detection", (int(self.up_ratio * im_w), int(text_scale * 20)),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, zone_color,
                        thickness=text_thickness)

            # 绘制检测区域-中线
            if self.det_mode == 0:
                zone_color = (130, 107, 220)  # 长春 BGR
                startPoint = (int(self.half_ratio * im_w), int(self.left_ratio * im_h))
                endPoint = (int(self.half_ratio * im_w), int(self.right_ratio * im_h))
                cv2.line(im, startPoint, endPoint, zone_color, thickness=2)

            # 绘制缓冲区域
            zone_color = (182, 182, 178)  # 月魄 BGR (182, 182, 178)
            cv2.rectangle(im, pt1=(int(self.buffer_up_ratio * im_w), int(self.buffer_left_ratio * im_h)),
                          pt2=(int(self.buffer_down_ratio * im_w), int(self.buffer_right_ratio * im_h)),
                          color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Buffer", (int(self.buffer_up_ratio * im_w), int(text_scale * 20)),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, zone_color,
                        thickness=text_thickness)

        return im

    # 对所有检测到的对象更新状态
    # now_timestamp: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.begin_time))
    def _update(self, obj_id, v_ratio, h_ratio, score, im, tlwh, now, frame_id):
        if not self.is_vertical:  # 不是竖直检测，则调换 vert_ratio和hori_ratio
            temp_ratio = v_ratio
            v_ratio = h_ratio
            h_ratio = temp_ratio

        # 如果不在容器内
        if not self.counter_dict.__contains__(obj_id):
            # 在检测区域内，才注册
            if self.up_ratio < v_ratio < self.down_ratio and self.left_ratio < h_ratio < self.right_ratio:
                # 在检测区域内，注册到容器，记录进入ratio
                begin_dir = self._get_begin_dir(v_ratio)
                # 如果方向不满足，返回 （不能这样写：对象迟早会进入另一半区，导致问题）
                # if self.only_dir != 1 and self.only_dir != begin_dir.value:
                #     return
                # print(f"register: {obj_id} begin_dir:{begin_dir}")
                self._register_item(obj_id, v_ratio, im, tlwh, now, begin_dir)
                # 更新抓拍信息（进入时）
                self._update_cap(obj_id, score, im, tlwh, now, self.border, frame_id=frame_id, v_ratio=v_ratio, h_ratio=h_ratio)
                # self.can_detect_face = True # 进入检测人脸
                print(f"{self.cam_name} 检测到对象进入检测区域: {obj_id} begin_dir:{begin_dir.name}")
                logger.info(f"{self.cam_name} 检测到对象进入检测区域: {obj_id} begin_dir:{begin_dir.name}")
            # 检测模式：里外包围盒。从外面进入，也需要生成对象，后续标记为无效
            elif self.det_mode == 1 and self.buffer_up_ratio < v_ratio < self.buffer_down_ratio and self.buffer_left_ratio < h_ratio < self.buffer_right_ratio:
                # 在检测区域内，注册到容器，记录进入ratio
                begin_dir = self._get_begin_dir(v_ratio)
                self._register_item(obj_id, v_ratio, im, tlwh, now, begin_dir)
                self.counter_dict[obj_id].set_det_flag()  # 设置无效状态
                print(f"{self.cam_name} 从缓冲区进入，为无效状态: {obj_id}")
                logger.info(f"{self.cam_name} 从缓冲区进入，为无效状态: {obj_id}")
                # 更新抓拍信息（进入时）
                self._update_cap(obj_id, score, im, tlwh, now, self.border, frame_id=frame_id, v_ratio=v_ratio, h_ratio=h_ratio)
        else:  # 在容器内
            item = self.counter_dict[obj_id]  # 拿到item
            if item.get_dying() == 0:  # 只有运行态可以更新
                # print(f"has been in the dict: {obj_id}")
                # 在缓冲区域内，更新状态
                if self.buffer_up_ratio < v_ratio < self.buffer_down_ratio and self.buffer_left_ratio < h_ratio < self.buffer_right_ratio:
                    self._update_cap(obj_id, score, im, tlwh, now, self.border, frame_id=frame_id, v_ratio=v_ratio, h_ratio=h_ratio)  # 更新状态（检测时）
                    # if item.get_trigger() == 1 and abs(v_ratio - self.half_ratio) < 0.02:
                    #     self.can_detect_face = True  # 中间检测人脸
                    #     self._update_cap(obj_id, score, im, tlwh, now, self.border, trigger=2)  # 更新状态（检测时），执行一次
                    # else:
                    #     self._update_cap(obj_id, score, im, tlwh, now, self.border, trigger=1)  # 更新状态（检测时）
                # 不在缓冲区域，计数，标记消亡态
                else:
                    # 更新最后一轮状态（检测时）
                    # if item.get_trigger() < 3: # 用一帧更新最后一轮状态
                    #     self._update_cap(obj_id, score, im, tlwh, now, self.border, trigger=3)
                    #     item.set_trigger(3)
                    #     return
                    # self.can_detect_face = True # 离开检测人脸
                    item.set_end_ratio(v_ratio)  # 设置离开ratio

                    if self.det_mode == 0:
                        dir = self.counter_dict[obj_id].get_dir()  # 得到最终方向
                        # 中途折返时返回Dir.Null，不计数
                        if dir != Dir.Null:
                            # 方向过滤
                            # print(f"dir.Null:{Dir.Null.value} dir.In:{Dir.In.value} dir.Out:{Dir.Out.value}")
                            if self.only_dir != 1 and self.only_dir != dir.value:  # 不满足，销毁对象
                                logger.info(f"{self.cam_name} {obj_id} 不满足方向性，剔除")
                                # print(f"only_dir:{self.only_dir} dir.value:{dir.value}")
                                item.clear_resource()
                                item.on_destroy()  # 销毁事件
                                del self.counter_dict[obj_id]  # 销毁对象
                            else:
                                fail_img = None if self.run_mode == 0 else self._get_track_img(im, tlwh, self.border)
                                self.on_sucess_quit(item, dir, now, fail_img)
                        else:  # 中途折返
                            logger.info(
                                f"{self.cam_name} 中途折返: {obj_id} begin_dir:{self.counter_dict[obj_id].begin_dir}")
                            item.clear_resource()
                            item.on_destroy()  # 销毁事件
                            del self.counter_dict[obj_id]  # 销毁对象
                    elif self.det_mode == 1:
                        if item.get_det_flag() == 0:  # 从检测区域里面到外面（有效）且只有可能是进或出
                            if self.only_dir == 2:
                                dir = Dir.In
                            else:
                                dir = Dir.Out

                            item.set_dir(dir) # 设置最终方向
                            fail_img = None if self.run_mode == 0 else self._get_track_img(im, tlwh, self.border)
                            self.on_sucess_quit(item, dir, now, fail_img)
                        else:  # 无效
                            item.clear_resource()
                            item.on_destroy()  # 销毁事件
                            del self.counter_dict[obj_id]  # 销毁对象
                            logger.info(
                                f"{self.cam_name} 无效，因为从外部缓冲区进入: {obj_id}")

            elif item.get_dying() == 1:  # 在将亡态又再次返回检测区域，需要销毁重新激活
                if self.up_ratio < v_ratio < self.down_ratio and self.left_ratio < h_ratio < self.right_ratio:
                    # 已经完成一轮计数，不用清理资源，销毁旧对象即可
                    self.counter_dict[obj_id].on_destroy()  # 销毁事件
                    del self.counter_dict[obj_id]  # 销毁对象
                    # 在检测区域内，注册到容器，记录进入ratio
                    begin_dir = self._get_begin_dir(v_ratio)
                    self._register_item(obj_id, v_ratio, im, tlwh, now, begin_dir)
                    if self.det_mode == 1:  # 里外包围盒模式：直接设为无效
                        self.counter_dict[obj_id].set_det_flag()
                    # 更新抓拍信息（进入时）
                    self._update_cap(obj_id, score, im, tlwh, now, self.border, frame_id=frame_id, v_ratio=v_ratio, h_ratio=h_ratio)

    # 成功离开
    def on_sucess_quit(self, item, dir, now, fail_img):
        # 成功离开事件
        item.on_success_quit(now, is_sql=self.is_sql,
                             qface_list=self.qface_list,
                             qsql_list=self.qsql_list,
                             fail_img=fail_img)

        if dir == Dir.In:
            self._in_count += 1
        else:
            self._out_count += 1
        if self.lzc_debug:
            print(f"{self.cam_name} 成功检测: {item.obj_id} {dir.name}")
            print(f"{self.cam_name} 当前 进入人数: {self._in_count} 离开人数:{self._out_count}")
            logger.info(f"{self.cam_name} 成功检测: {item.obj_id} {dir.name}")
            logger.info(
                f"{self.cam_name} 当前 进入人数: {self._in_count} 离开人数:{self._out_count}")

        if self.is_vid:
            item.set_dying(1)  # 设为消亡态
        else:
            item.on_destroy()  # 销毁事件
            del self.counter_dict[item.obj_id]  # 销毁对象

    # 定时函数（每帧执行逻辑前调用）
    def schedule_before(self, now, frame_id):
        for key in list(self.counter_dict.keys()):
            item = self.counter_dict[key]
            if item.get_dying() == 0 and frame_id - item.last_frame_id > self.lost_frame:
                logger.info(f"{self.cam_name} 对象消失超过{self.lost_frame}帧，自动销毁: {key}")
                print(f"{self.cam_name} 对象消失超过{self.lost_frame}帧，自动销毁: {key}")
                self.counter_dict[key].clear_resource()  # 清理相关数据
                self.counter_dict[key].on_destroy()  # 正常退出
                del self.counter_dict[key]  # 销毁对象

    # 定时函数（每帧执行逻辑后调用）
    def schedule(self, im, now):
        # 每帧执行(是否继续录制视频)
        if self.is_vid:
            for key in list(self.counter_dict.keys()):
                item = self.counter_dict[key]
                if item.get_dying() == 1:  # item处于将亡态
                    item.record_video(im, now)
                    item.click_save_fps()
                elif self.counter_dict[key].get_dying() == 2:
                    item.on_destroy()  # 销毁事件
                    del self.counter_dict[key]  # 销毁对象

        # 每10秒执行（如果可以）
        if now - self._last_schedule_sec1 > 10:
            self._last_schedule_sec1 = now
            self._destroy_unused(now)  # 清理无用对象

        # 每24h执行
        if now - self._last_schedule_hour1 > 60 * 60 * 24:
            self._last_schedule_hour1 = now
            self._in_count = 0
            self._out_count = 0

    # 清理无用资源（当硬件问题导致追踪目标丢失时，可能导致内存泄露，需要定期清理内存）
    def _destroy_unused(self, now):
        for key in list(self.counter_dict.keys()):
            if now - self.counter_dict[key].get_update_time() > self._destroy_time:
                logger.info(f"{self.cam_name} 对象消失过久，强制销毁: {key}")
                print(f"{self.cam_name} 对象消失过久，强制销毁: {key}")
                self.counter_dict[key].clear_resource()  # 清理相关数据
                self.counter_dict[key].on_destroy()  # 正常退出
                del self.counter_dict[key]  # 销毁对象

            # if now - self.counter_dict[key].begin_time > self._black_time:
            #     logger.info(f"{self.cam_name} 对象停留时间过久，自动销毁: {key}")
            #     print(f"{self.cam_name} 对象停留时间过久，自动销毁: {key}")
            #     self.counter_dict[key].clear_resource()  # 清理相关数据
            #     self.counter_dict[key].on_destroy()  # 正常退出
            #     del self.counter_dict[key]  # 销毁对象
            #     continue

    # 更新抓拍信息
    # trigger 1:进入区域 2:在区域内 3:离开区域
    def _update_cap(self, obj_id, score, im, tlwh, now, border=1, frame_id=0, v_ratio=0, h_ratio=0):
        item = self.counter_dict[obj_id]
        # 更新时间
        item.set_update_time(now)
        # 更新帧率
        item.last_frame_id = frame_id
        # 更新now_ratio
        item.now_v_ratio = v_ratio
        # 里外包围盒模式，无效情况
        if item.get_det_flag() == 1:
            return

        # 抓拍图片
        if self.is_img:
            # if self.counter_dict[obj_id].check_best_score(score):
            #     self.counter_dict[obj_id].set_best_img(self._get_face_img(im, tlwh, border))
            # 保存置信度高的图片
            if self.run_mode == 0:
                if item.check_best_score(score):
                    item.set_best_img(self._get_track_img(im, tlwh, border))
            elif self.run_mode == 1:  # 人脸需要二次检测
                # if item.get_trigger() < trigger:
                #     # 人脸检测
                #     item.set_trigger(trigger)
                # ltrb
                # 扩大头像范围
                l = tlwh[0] - border
                t = tlwh[1] - border
                r = tlwh[0] + tlwh[2] + border
                b = tlwh[1] + tlwh[3] + border
                if l < 0: l = 0
                if t < 0: t = 0
                if r > im.shape[1]: r = im.shape[1]
                if b > im.shape[0]: b = im.shape[0]
                item.ltrb = (l, t, r, b)
                item.is_match = False  # 刷新匹配标识

                # 如果没有成功抓拍就传入人的抓拍图
                if not item.get_is_success_img():
                    if item.check_frame_counter(v_ratio, h_ratio):
                        item.put_img(self._get_track_img(im, tlwh, border=0), y_ratio=tlwh[1] / im.shape[0])

        # 录制视频
        if self.is_vid:
            item.record_video(im, now)

    def _get_track_img2(self, im, ltrb, border=1):
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

    def _register_item(self, obj_id, now_ratio, im, tlwh, now, begin_dir):
        self.counter_dict[obj_id] = MyCoutnerItem(
            {
                "obj_id": obj_id,
                "cam_id": self.cam_id,
                "run_mode": self.run_mode,
                "begin_ratio": now_ratio,
                "begin_dir": begin_dir,
                "begin_time": now,
                "is_reverse": self.is_reverse,
                # "up_ratio": self.up_ratio,
                # "down_ratio": self.down_ratio,
                "save_path": self.save_path,
                "is_img": self.is_img,
                "is_vid": self.is_vid,
                "save_fps": self.save_fps,
                "cap_fps": self.cap_fps,
                "cap_width": self.cap_width,
                "cap_height": self.cap_height,
                "cam_name": self.cam_name,
                "conf": self.conf,
                "per_img_count": self.per_img_count,
                "per_img_interval": self.per_img_interval,
            }
        )
        # 生成一张候选图
        self.counter_dict[obj_id].put_img(self._get_track_img(im, tlwh, border=0))

    def _get_begin_dir(self, begin_ratio):
        # 标记初始ratio所在区域，用于方向双重认证
        if self.up_ratio <= begin_ratio <= self.half_ratio:
            # print(f"Dir In: {self.obj_id}")
            begin_dir = Dir.In
            if self.is_reverse:
                begin_dir = Dir.Out
        else:
            # print(f"Dir Out: {self.obj_id}")
            begin_dir = Dir.Out
            if self.is_reverse:
                begin_dir = Dir.In
        return begin_dir


# 生成counterMgr
def create_counterMgr(main_yaml, cam_yaml, output_dir, qface_list, qsql_list,
                      face_predictor=None, test_size=(640, 640)):
    counterMgr = MyCounterMgr(
        {
            "run_mode": cam_yaml['run_mode'],
            "cam_name": cam_yaml['cam_name'],
            "cal_mode": cam_yaml['cal_mode'],
            "det_mode": cam_yaml['det_mode'],
            "cam_id": cam_yaml['cam_id'],
            "center_ratio": cam_yaml['center_ratio'],
            "up_ratio": cam_yaml['zone'][0],
            "down_ratio": cam_yaml['zone'][1],
            "buffer_up_ratio": cam_yaml['zone'][2],
            "buffer_down_ratio": cam_yaml['zone'][3],
            "left_ratio": cam_yaml['zone'][4],
            "right_ratio": cam_yaml['zone'][5],
            "buffer_left_ratio": cam_yaml['zone'][6],
            "buffer_right_ratio": cam_yaml['zone'][7],
            "is_reverse": cam_yaml['is_reverse'],
            "is_vertical": cam_yaml['is_vertical'],
            "only_dir": cam_yaml['only_dir'],
            "lzc_debug": main_yaml['is_debug'] and cam_yaml['is_debug'],
            # "face_interval": main_yaml['face']['detect_interval'],
            "is_sql": main_yaml['mysql']['is_sql'],
            "border": cam_yaml['border'],
            "lost_frame": cam_yaml['lost_frame'],
            "is_img": cam_yaml['is_img'],
            "is_vid": cam_yaml['cap_vid']['is_vid'],
            "cap_width": cam_yaml['cap_vid']['width'],
            "cap_height": cam_yaml['cap_vid']['height'],
            "cap_fps": cam_yaml['cap_vid']['fps'],
            "save_fps": cam_yaml['cap_vid']['save_fps'],
            "save_path": output_dir,
            "qface_list": qface_list,
            "qsql_list": qsql_list,
            "face_predictor": face_predictor,
            "test_size": test_size,
            "is_vis": cam_yaml['is_vis'],
            "conf": 0.67 if cam_yaml['run_mode'] == 0 else cam_yaml['args2']['conf'],
            "per_img_count": cam_yaml['per_img_count'],
            "per_img_interval": cam_yaml['per_img_interval'],
        }
    )
    return counterMgr
