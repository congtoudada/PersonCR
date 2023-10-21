import sys
import cv2
import numpy as np

from queue import Queue
from multiprocessing import Manager
from typing import Tuple
from loguru import logger
from tools.lzc.count.framework.CountDrawTool import CountDrawTool
from tools.lzc.count.framework.CountEnum import ZoneEnum, CountItemStateEnum
from tools.lzc.count.framework.CountItem import CountItem
from tools.lzc.count.framework.CountMgrData import CountMgrData, CountMgrRunningData
from tools.lzc.count.framework.CountPool import CountPool
from tools.lzc.count.framework.ICountMgr import ICountMgr
from tools.lzc.count.framework.ScheduleTask import ScheduleTask
from tools.lzc.face.framework.FaceRegTool import FaceRegTool


class CountMgr(ICountMgr):
    def __init__(self, init_dict):
        # 内部初始化
        self.in_count = 0  # 进人数
        self.out_count = 0  # 出人数
        self.count_items: [int, CountItem] = {}  # count_item的容器
        self.data = CountMgrData(init_dict)
        # self.count_item_pool = CountPool(10, CountItem if self.data.run_mode == 0 else FaceRegItem)
        self.count_item_pool = CountPool(10, CountItem)
        self.running_data = CountMgrRunningData()
        self.frame_history = Queue(maxsize=self.data.frame_history_capacity)
        self.max_int = int(sys.maxsize * 0.999)
        # 5*60s打印一次CountMgr状态
        self.debug_info_task = ScheduleTask(5 * 60, -1, lambda: logger.info(self.debug_info()))

        # 使用外部参数
        self._init(self.data)

    def _init(self, data):
        self.detect_left_up = (data.left_ratio, data.up_ratio)
        self.detect_right_down = (data.right_ratio, data.down_ratio)
        self.buffer_left_up = (data.buffer_left_ratio, data.buffer_up_ratio)
        self.buffer_right_down = (data.buffer_right_ratio, data.buffer_down_ratio)
        if data.is_vertical:
            self.half_ratio = data.up_ratio * (1 - data.center_ratio) + data.down_ratio * data.center_ratio
            self.buffer_half_ratio = data.buffer_up_ratio * (
                    1 - data.center_ratio) + data.buffer_down_ratio * data.center_ratio
        else:
            self.half_ratio = data.left_ratio * (1 - data.center_ratio) + data.right_ratio * data.center_ratio
            self.buffer_half_ratio = data.buffer_left_ratio * (
                    1 - data.center_ratio) + data.buffer_right_ratio * data.center_ratio

    def get_container(self) -> dict:
        return self.count_items

    # 计数更新 (由外部每帧调用)
    def update(self, im, tlwhs, ids, scores, now, frame_id):
        im = np.ascontiguousarray(np.copy(im))
        im_h, im_w = im.shape[:2]
        if self.frame_history.full():
            self.frame_history.get()
        self.frame_history.put(im)

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            obj_id = int(ids[i])
            # 追踪对象状态更新
            data = self.data
            if data.cal_mode == 0:  # 左上角检测
                self._item_update(obj_id, float(y1 / im_h), float(x1 / im_w), scores[i], im, tlwh,
                                  now=now, frame_id=frame_id)
                if data.is_vis:
                    cv2.circle(im, (int(x1), int(y1)), 4, (118, 154, 242), 2)
            else:  # 中心检测
                self._item_update(obj_id, float((y1 + 0.5 * h) / im_h), float((x1 + 0.5 * w) / im_w), scores[i], im,
                                  tlwh, now=now, frame_id=frame_id)
                if data.is_vis:
                    cv2.circle(im, (int(x1 + 0.5 * w), int(y1 + 0.5 * h)), 4, (118, 154, 242), 2)
        return im

    # global_update
    def global_update(self, im, now, frame_id):
        self._process_reg_rsp()  # 处理人脸识别结果
        self.debug_info_task.schedule_update(now)  # 定时回调:打印debug信息

        # 资源管理
        for key in list(self.count_items.keys()):
            item: CountItem = self.count_items[key]
            if item.state == CountItemStateEnum.Dead:
                self._item_recycle(item, is_clear=False)
            else:
                item.global_update(im, now, frame_id)

            # 长时间未更新，直接释放
            if frame_id - item.last_update_frame > self.data.lost_frames:
                logger.info(f"{self.data.cam_name} 对象消失超过{self.data.lost_frames}帧，自动销毁: {key}")
                self._item_recycle(item, is_clear=True)

    # 绘制可视化信息
    def draw(self, im):
        if self.data.is_vis:
            CountDrawTool.draw_count(self, im)
            CountDrawTool.draw_zone(self, im)
        return im

    def debug_info(self) -> str:
        return f"当前CountMgr状态: \n" \
               f"count_items.size: {self.count_items.__len__()}\n" \
               f"count_item_pool.size: {self.count_item_pool.pool.__len__()}"

    def print_params_info(self):
        print('-----------------------begin params----------------------------------')
        # 遍历params中的每一个键值对，添加到表格中
        for key, value in self.data.init_dict.items():
            print(f"| {key} : {value} |")

        print('-----------------------end params----------------------------------')

    # 计数核心逻辑(每帧没对象调用)
    def _item_update(self, obj_id, v_ratio, h_ratio, score, im, tlwh, now, frame_id):
        point = (h_ratio, v_ratio)
        self.running_data.init(obj_id, point, score, im, self.frame_history.queue[0], tlwh, now, frame_id)
        data: CountMgrData = self.data
        running_data: CountMgrRunningData = self.running_data

        # 如果不在容器内
        if not self.count_items.__contains__(obj_id):
            # 首次进入检测区域
            self.on_start(data, running_data)
        else:  # 在容器内
            count_item: CountItem = self.count_items[obj_id]  # 拿到count_item
            self.on_update(count_item, data, running_data)

    def on_start(self, data: CountMgrData, running_data: CountMgrRunningData):
        # 不在容器，首次进入检测区域:
        #   常规模式: 方向过滤，Valid或Invalid 加入容器
        #   里外包围盒模式：Valid加入容器
        if self._check_in_detect(running_data.point):
            # 在检测区域内，注册到容器，记录进入zone，标记为Valid
            begin_zone = self._get_zone(running_data.point, is_detect=True)
            begin_state = CountItemStateEnum.Valid

            # 常规模式, 过滤方向
            if data.det_mode == 0:
                filter_zone = ZoneEnum.Null  # 需要过滤掉的方向
                if data.only_dir == 2:
                    filter_zone = ZoneEnum.Out
                elif data.only_dir == 3:
                    filter_zone = ZoneEnum.In

                if begin_zone == filter_zone:
                    begin_state = CountItemStateEnum.Invalid

            self._register_item(running_data, begin_zone, begin_state)

            if begin_state.Valid:
                logger.info(
                    f"{data.cam_name} Valid对象进入检测区域: {running_data.obj_id} begin_zone:{begin_zone.name}")
            else:
                logger.info(
                    f"{data.cam_name} Invalid对象进入检测区域: {running_data.obj_id} begin_zone:{begin_zone.name}")
        # 不在容器内，首次进入缓冲区域:
        #   常规模式：无
        #   里外包围盒：Invalid加入容器
        elif self._check_in_buffer(running_data.point):
            # 不在容器内，首次进入缓冲区域
            #   常规模式：无
            #   里外包围盒：Invalid加入容器
            if data.det_mode == 1:
                if self._check_in_rect(running_data.point, self.buffer_left_up, self.buffer_right_down):
                    self._register_item(running_data, ZoneEnum.Null, CountItemStateEnum.Invalid)
                    logger.info(f"{data.cam_name} 从缓冲区进入，为无效状态: {running_data.obj_id}")

    def on_update(self, count_item: CountItem, data: CountMgrData, running_data: CountMgrRunningData):
        # 有效状态或不合法状态才更新并计数
        if count_item.state == CountItemStateEnum.Valid:
            # 合法对象在缓冲区域内
            if self._check_in_buffer(running_data.point):
                count_item.buffer_update_valid(data, running_data)  # 合法item更新
            # 合法对象在缓冲区域外
            else:
                if data.det_mode == 0:  # 常规模式，根据方向判断计数or回收
                    end_zone = self._get_zone(running_data.point, is_detect=False)
                    if end_zone != count_item.begin_zone:  # 进出标志不同，结果有效
                        self._item_success_quit(count_item, data, running_data)  # 触发成功计数事件
                    else:
                        logger.info(
                            f"{data.cam_name} 中途折返，不计数: {count_item.obj_id} begin_zone:{count_item.begin_zone}")
                        self._item_recycle(count_item, is_clear=True)  # 强制回收并清空资源
                elif data.det_mode == 1:  # 里外包围盒模式，计数
                    self._item_success_quit(count_item, data, running_data)  # 触发成功计数事件
        elif count_item.state == CountItemStateEnum.Invalid:
            # 不合法对象在缓冲区域内
            if self._check_in_buffer(running_data.point):
                count_item.buffer_update_invalid(data, running_data)  # 不合法item更新
            # 不合法对象在缓冲区域外
            else:
                logger.info(
                    f"{data.cam_name} 不合法对象离开缓冲区，销毁: {count_item.obj_id} begin_zone:{count_item.begin_zone}")
                self._item_recycle(count_item, is_clear=True)  # 强制回收并清空资源
        # 将亡状态再次返回检测区域,终止录制并回收
        elif count_item.state == CountItemStateEnum.Dying:
            # item.buffer_update更新不在这，避免将亡态物体没有追踪到就不更新了
            # item.buffer_update(data, running_data)
            # 如果再次返回检测区域，则立即停止录制并回收（不清空资源）
            if self._check_in_rect(running_data.point, self.detect_left_up, self.detect_right_down):
                self._item_recycle(count_item, is_clear=False)

    # 成功离开
    def _item_success_quit(self, item, data: CountMgrData, running_data: CountMgrRunningData):
        item.on_success_quit(data, running_data)
        # 计数
        if item.begin_zone == ZoneEnum.In:
            self.in_count += 1
            if self.in_count > self.max_int:
                self.in_count = 0
        else:
            self.out_count += 1
            if self.out_count > self.max_int:
                self.out_count = 0

        logger.info(f"{self.data.cam_name} 成功计数! obj_id: {item.obj_id} begin_zone:{item.begin_zone}")
        if (self.in_count + self.out_count) % 10 == 0 or self.data.debug_mode:
            logger.info(f"{self.data.cam_name} 当前 进入人数: {self.in_count} 离开人数:{self.out_count}")

    # 检测是否在检测区域内
    def _check_in_detect(self, point: Tuple[int, int]):
        return self._check_in_rect(point, self.detect_left_up, self.detect_right_down)

    # 检测是否在缓冲区域内
    def _check_in_buffer(self, point: Tuple[int, int]):
        return self._check_in_rect(point, self.buffer_left_up, self.buffer_right_down)

    # 检测是否在矩形框内
    def _check_in_rect(self, point: Tuple[int, int], rect_left_up: Tuple[int, int],
                       rect_right_down: Tuple[int, int]) -> bool:
        return rect_left_up[0] < point[0] < rect_right_down[0] and rect_left_up[1] < point[1] < rect_right_down[1]

    # 检测区域标号 (is_detect为Treu：以检测区域计算；为False: 以Buffer区域计算)
    # 举例：竖直检测，不反向。默认往上代表离开，因此下半矩形为Out区域，上半矩形为In (Zone也可以理解成方向，先进哪个区域就代表走哪个方向）
    def _get_zone(self, point: Tuple[int, int], is_detect):
        data = self.data
        zone = ZoneEnum.Null
        if data.det_mode == 0:
            if is_detect:
                rect_right_down = self.detect_right_down
                half_ratio = self.half_ratio
            else:
                rect_right_down = (1, 1)
                half_ratio = self.buffer_half_ratio

            if data.is_vertical:
                # 不考虑反向，如果处于下半区域，代表离开
                if half_ratio < point[1] < rect_right_down[1]:
                    zone = ZoneEnum.Out
                else:
                    zone = ZoneEnum.In
            else:
                # 不考虑反向，如果处于右半区域，代表离开
                if half_ratio < point[0] < rect_right_down[0]:
                    zone = ZoneEnum.Out
                else:
                    zone = ZoneEnum.In
        elif data.det_mode == 1:
            if is_detect:
                zone = ZoneEnum.In if data.only_dir == 2 else ZoneEnum.Out

        if data.is_reverse:
            if zone == ZoneEnum.In:
                zone = ZoneEnum.Out
            elif zone == ZoneEnum.Out:
                zone = ZoneEnum.In
        return zone

    # 申请count_item到容器
    def _register_item(self, running_data, begin_zone, state):
        count_item: CountItem = self.count_item_pool.pop()
        count_item.init(self, running_data, begin_zone, state)
        self.count_items[running_data.obj_id] = count_item
        return count_item

    # 处理人脸识别响应
    def _process_reg_rsp(self):
        if self.data.run_mode != 1:
            return

        rsp_list = self.data.qface_rsp
        if rsp_list:
            while not rsp_list.empty():
                obj_id, per_id, score, face_img = FaceRegTool.unpack_rsp(rsp_list.get())
                if self.count_items.__contains__(obj_id):
                    self.count_items[obj_id].process_reg_rsp(per_id, score, face_img)

    # 回收对象，是否连同清除该对象生成的资源
    def _item_recycle(self, item, is_clear):
        item.on_release(is_clear)  # 通知资源释放
        del self.count_items[item.obj_id]  # 从容器内移除
        self.count_item_pool.push(item)  # 将对象放回缓存池

    @staticmethod
    def allocate(main_yaml, cam_yaml, output_dir, qface_req, qface_rsp, qsql_list) -> ICountMgr:
        return CountMgr(
            {
                # cam_yaml
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
                "is_vertical": cam_yaml['is_vertical'],
                "is_reverse": cam_yaml['is_reverse'],
                "only_dir": cam_yaml['only_dir'],
                "is_vis": cam_yaml['is_vis'],
                # 抓拍设置
                "is_img": cam_yaml['cap']['is_img'],
                "border": cam_yaml['cap']['border'],
                "is_vid": cam_yaml['cap']['is_vid'],
                "cap_width": cam_yaml['cap']['width'],
                "cap_height": cam_yaml['cap']['height'],
                "cap_fps": cam_yaml['cap']['fps'],
                "save_frames": cam_yaml['cap']['save_frames'],
                "max_record_frames": cam_yaml['cap']['max_record_frames'],
                "lost_frames": cam_yaml['cap']['lost_frames'],
                # main_yaml
                "debug_mode": main_yaml['debug_mode'],
                "frame_history_capacity": main_yaml['cam']['frame_history_capacity'],
                "reg_interval": main_yaml['cam']['reg_interval'],
                "reg_count_thresh": main_yaml['cam']['reg_count_thresh'],
                # 其他
                "save_path": output_dir,
                "qface_req": qface_req,
                "qface_rsp": qface_rsp,
                "qsql_list": qsql_list
            }
        )
