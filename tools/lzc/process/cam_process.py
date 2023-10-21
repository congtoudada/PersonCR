import argparse
import os
import os.path as osp
import queue
import random
import sys
import time
import traceback
from multiprocessing import current_process, Pipe, Manager

import cv2
import torch

from loguru import logger

from tools.lzc.config_tool import ConfigTool
from tools.lzc.count.framework.CountMgr import CountMgr
from tools.lzc.count.framework.ICountMgr import ICountMgr
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from tools.lzc.Predictor import create_predictor, Predictor


def _override_args(args, cam_yaml):
    # 如果可以，用配置文件重载args配置
    args.expn = cam_yaml['args1']['expn']
    args.path = cam_yaml['args1']['path']
    args.save_result = cam_yaml['args1']['save_result']
    args.exp_file = cam_yaml['args1']['exp_file']
    args.ckpt = cam_yaml['args1']['ckpt']
    args.conf = cam_yaml['args1']['conf']
    args.fps = cam_yaml['args1']['fps']
    args.track_thresh = cam_yaml['args1']['track_thresh']
    args.track_buffer = cam_yaml['args1']['track_buffer']
    args.match_thresh = cam_yaml['args1']['match_thresh']
    args.aspect_ratio_thresh = cam_yaml['args1']['aspect_ratio_thresh']
    return args


# 多核进程
# 向共享缓冲栈中写入数据，配合read_process使用
def write_process(qframe, cam_event, esc_event, main_yaml, cam_yaml) -> None:
    pname = f"[ {os.getpid()}:{cam_yaml['cam_name']} Write ]"

    # 使用spawn会创建全新进程，需重新配置日志
    ConfigTool.load_log_config(main_yaml)

    # 准备好可能需要的变量
    cam_url = cam_yaml['args1']['path']  # 取流地址
    debug_mode = main_yaml['debug_mode']
    jump_thresh = main_yaml['cam']['jump']  # 3
    drop_interval = main_yaml['cam']['drop_interval']  # 2: 每2帧里面丢1帧

    while not cam_event.is_set():
        time.sleep(0.2)  # 等待读进程初始化

    # 等待处理进程初始化后，开始初始化并接收视频流
    cap = cv2.VideoCapture(cam_url)
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    is_cal_frame = False
    real_frame = 0
    drop_flag = 0

    logger.info(f"{pname} work! video stream from: {cam_url}!")
    while True:
        try:
            if cap.isOpened():
                if qframe.full():  # 队列满，被动丢帧
                    if real_frame % 20 == 0:
                        logger.error(f"{pname} 服务器存在性能瓶颈，无法正常读取并处理视频流！")
                        cap.grab()
                else:
                    # 自适应主动丢帧
                    now_jump = int(qframe.qsize() / jump_thresh)
                    if now_jump > 0:
                        logger.warning(f"{pname} 服务器存在性能瓶颈，自适应丢帧: {now_jump}！")
                    for i in range(now_jump):
                        cap.grab()

                    # 主动丢帧
                    drop_flag += 1
                    if drop_flag >= drop_interval:
                        drop_flag = 0
                        cap.grab()
                    else:
                        status, frame = cap.read()
                        if status:
                            # print("resize before:", frame.shape)
                            # cv2.imsize(frame, (1920, 1080))
                            # print("resize after:", frame.shape)
                            qframe.put(frame)
                            # real_frame += 1

                # if real_frame % 200 == 0 and is_cal_frame:
                #     print("video stream:", frame.shape)
                #     logger.info(f"{pname} real frame num: {real_frame}. queue.len: {qframe.qsize()}")
            else:
                logger.error(f"{pname} video stream closed, Try get video stream!")
                time.sleep(1)
                cap = cv2.VideoCapture(cam_url)
        except Exception as e:
            logger.info(f"{pname} {traceback.format_exc()}")
            cap = cv2.VideoCapture(cam_url)

        if debug_mode:
            time.sleep(1.0 / frame_fps)  # 读本地文件要用该行
        if esc_event.is_set():
            logger.info(f"{pname} Exit!")
            return


# 在缓冲栈中读取数据，配置write_process使用
def read_process(qframe, cam_event, qface_req, qface_rsp, qsql_list, esc_event, args, main_yaml,
                 cam_yaml) -> None:
    pname = f"[ {os.getpid()}:{cam_yaml['cam_name']} Read ]"

    # 使用spawn会创建全新进程，需重新配置日志
    ConfigTool.load_log_config(main_yaml)

    # ---------------------------- 准备所需变量 ----------------------------
    if main_yaml['enable_args']:
        args = _override_args(args, cam_yaml)

    run_mode = cam_yaml['run_mode']
    exp = get_exp(args.exp_file, args.name)
    # 如果没有取实验名，就取yolox_s
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # 设置输出文件夹
    video_name = args.path.split('/')[-1].split('.')[0]
    # output_dir: YOLOX_outputs/renlian/test01/
    output_dir = osp.join(exp.output_dir, args.expn, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # yolox检测器
    predictor = create_predictor(args, exp)

    # 追踪器初始化
    tracker = BYTETracker(args, frame_rate=args.fps)
    current_time = time.localtime()
    # 获取视频流及其参数后释放
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    # save_path: YOLOX_outputs/renlian/test01/test01.mp4
    save_path = osp.join(output_dir, f"{osp.basename(output_dir)}.mp4")

    vid_writer = None
    if args.save_result:
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    timer = Timer()
    frame_id = 0

    # 自定义
    run_mode = cam_yaml['run_mode']
    debug_mode = main_yaml['debug_mode']
    is_vis = cam_yaml['is_vis']
    face_infer_flag = 0
    match_list = []

    countMgr: ICountMgr = CountMgr.allocate(main_yaml, cam_yaml, output_dir, qface_req, qface_rsp, qsql_list)

    logger.info(f"{pname} work!")
    if debug_mode:
        countMgr.print_params_info()
    cam_event.set()


    # ---------------------------- 工作循环 ----------------------------
    while True:
        # try:
        if not qframe.empty():
            frame = qframe.get()
            now = time.time()

            # DEBUG打印
            if debug_mode and frame_id % 20 == 0:
                logger.info(
                    f"{pname} : Processing frame {frame_id} with {1. / max(1e-5, timer.average_time):.2f} fps")
                # timer.clear()

            # 目标检测(检测人)
            outputs, img_info = predictor.inference(frame, timer)

            if outputs[0] is not None:
                outputs[0] = outputs[0][outputs[0][:, 6] == 0, :]  # 都先追踪人，保留boundingbox

                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']],
                                                exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                timer.toc()

                # Update
                online_im = countMgr.update(img_info['raw_img'], online_tlwhs, online_ids, online_scores, now,
                                            frame_id)

                # 画监测区域（由内部判断是否画）
                online_im = countMgr.draw(online_im)

                # 可视化线框
                if is_vis:
                    online_im = plot_tracking(
                        online_im, online_tlwhs, online_ids, scores=online_scores, frame_id=frame_id + 1,
                        fps=1. / timer.average_time, per_ids=None if run_mode == 0 else countMgr.get_container()
                    )  # 画追踪boundingbox
            else:
                timer.toc()
                online_im = img_info['raw_img']
                if frame_id > int(sys.maxsize * 0.999):
                    frame_id = 0

            # global Update
            countMgr.global_update(img_info['raw_img'], now, frame_id)

            if args.save_result:
                vid_writer.write(online_im)

            frame_id += 1

        # except Exception as e:
        #     logger.info(f"{pname} {e}")

        if esc_event.is_set():
            if vid_writer is not None:
                vid_writer.release()
            logger.info(f"{pname} Exit!")
            return
