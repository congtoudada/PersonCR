import argparse
import os
import os.path as osp
import queue
import random
import sys
import time
from multiprocessing import current_process, Pipe, Manager

import cv2
import torch

from loguru import logger
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from tools.lzc.counter_mgr import MyCounterMgr, create_counterMgr
from tools.lzc.Predictor import create_predictor, Predictor
from tools.lzc.my_logger import log_config


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
    print(f'{pname} launch!')
    log_config(main_id=main_yaml['main_id'])

    # 准备好可能需要的变量
    cam_url = cam_yaml['args1']['path']

    while not cam_event.is_set():
        time.sleep(0.1)  # 等待读进程初始化

    # 等待处理进程初始化后，开始初始化并接收视频流
    cap = cv2.VideoCapture(cam_url)
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    is_cal_frame = False
    real_frame = 0
    jump_thresh = 3
    drop_interval = 2  # 每3帧丢1帧
    drop_flag = 0

    logger.info(f"{pname} work! video stream from: {cam_url}!")
    print(f"{pname} work! video stream from: {cam_url}!")
    while True:
        # time.sleep(1.0 / frame_fps)  # 读本地文件要用该行
        if esc_event.is_set():
            print(f"{pname} Exit!")
            return

        try:
            if cap.isOpened():
                if qframe.full():  # 队列满，被动丢帧
                    if real_frame % 20 == 0:
                        logger.error(f"{pname} 服务器存在性能瓶颈，无法正常读取并处理视频流！")
                        print(f"{pname} 服务器存在性能瓶颈，无法正常读取并处理视频流！")
                        cap.grab()
                    # if cap.grab():
                    #     real_frame += 1
                else:
                    # 自适应主动丢帧
                    now_jump = int(qframe.qsize() / jump_thresh)
                    for i in range(now_jump):
                        cap.grab()
                        logger.error(f"{pname} 服务器存在性能瓶颈，自适应丢帧！")
                        print(f"{pname} 服务器存在性能瓶颈，自适应丢帧！")
                        # if cap.grab():
                        #     real_frame += 1

                    # 主动丢帧
                    drop_flag += 1
                    if drop_flag >= drop_interval:
                        drop_flag = 0
                        cap.grab()
                        # if cap.grab():
                        #     real_frame += 1
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
                print(f"{pname} video stream closed, Try get video stream!")
                cap = cv2.VideoCapture(cam_url)
                frame_fps = cap.get(cv2.CAP_PROP_FPS)
                return
        except Exception as e:
            logger.info(f"{pname} {e}")
            print(f"{pname} {e}")
            cap = cv2.VideoCapture(cam_url)
            frame_fps = cap.get(cv2.CAP_PROP_FPS)

# 在缓冲栈中读取数据，配置write_process使用
def read_process(qframe, cam_event, qface_list, qsql_list, esc_event, args, main_yaml, cam_yaml) -> None:
    pname = f"[ {os.getpid()}:{cam_yaml['cam_name']} Read ]"
    print(f'{pname} launch!')
    log_config(main_id=main_yaml['main_id'])

    # ---------------------------- 准备所需变量 ----------------------------
    if main_yaml['is_args']:
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

    # 人检测器
    predictor = create_predictor(args, exp)

    # 人脸检测器
    face_predictor = None
    test_size = None
    if main_yaml['is_args'] and run_mode == 1:
        args.exp_file = cam_yaml['args2']['exp_file']
        args.ckpt = cam_yaml['args2']['ckpt']
        args.conf = cam_yaml['args2']['conf']
        args.nms = cam_yaml['args2']['nms']
        exp2 = get_exp(args.exp_file, args.name)
        test_size = exp2.test_size
        face_predictor = create_predictor(args, exp2)

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
    lzc_debug = cam_yaml['is_debug'] and main_yaml['is_debug']  # 是否打印debug信息
    is_vis = cam_yaml['is_vis']
    border = cam_yaml['border']
    is_cal_frame = True
    face_outputs = None
    face_infer_flag = 0
    match_list = []

    counterMgr = create_counterMgr(main_yaml, cam_yaml, output_dir, qface_list, qsql_list,
                                   face_predictor=face_predictor, test_size=test_size)

    cam_event.set()
    logger.info(f"{pname} work!")
    print(f"{pname} work!")

    # ---------------------------- 工作循环 ----------------------------
    while True:
        try:
            if not qframe.empty():
                # if qframe.qsize() > 1:
                frame = qframe.get()
                now = time.time()

                # 打印（可删）
                # if frame_id % 90000 == 0 and is_cal_frame:
                #     # logger.info(
                #     #     f"{pname} : Processing frame {frame_id} with {1. / max(1e-5, timer.average_time):.2f} fps")
                #     # timer.clear()
                #     cv2.imwrite(f"./temp/{pname}_{frame_id}.jpg", frame)

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

                    # 预更新
                    counterMgr.schedule_before(now, frame_id)

                    # 画监测区域
                    # counterMgr.can_detect_face = False
                    online_im = counterMgr.update(img_info['raw_img'], online_tlwhs, online_ids, online_scores, now, frame_id)

                    online_im = counterMgr.draw_zone(online_im)
                    # 可视化线框
                    if is_vis:
                        online_im = plot_tracking(
                            online_im, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                        )  # 画追踪boundingbox

                    # 目标检测(检测人脸)，根据脸在人内的先验，匹配人脸
                    if run_mode == 1:
                        face_infer_flag += 1
                        if face_infer_flag % 3 == 0:
                            face_infer_flag = 0
                            if len(counterMgr.counter_dict) > 0: # 存在检测对象，检测人脸
                                # if counterMgr.can_detect_face:
                                face_outputs, face_img_info = face_predictor.inference(frame, timer)
                                if face_outputs[0] is not None:
                                    output_results = face_outputs[0]
                                    bboxes = output_results[:, :4]  # x1y1x2y2
                                    scale = min(exp.test_size[0] / float(face_img_info['height']), exp.test_size[1] / float(face_img_info['width']))
                                    bboxes /= scale
                                    scores = output_results[:, 4] * output_results[:, 5]

                                    bboxes = bboxes.cpu().tolist()
                                    scores = scores.cpu().tolist()
                                    for_count = face_outputs[0].shape[0]
                                    # 根据bboxes的y轴坐标做脸的降序排序
                                    # sorted_index_list = sorted(range(len(bboxes)), key=lambda x: bboxes[x][1], reverse=True)

                                    # print(f"for_count: {for_count}")
                                    match_list.clear()
                                    for key in counterMgr.counter_dict.keys():
                                        if counterMgr.counter_dict[key].get_dying() == 0 and counterMgr.counter_dict[key].get_det_flag() == 0:  # 处于运行态的才更新
                                            match_list.append(key)

                                    # match_list.sort() # 根据y轴做人的降序排序
                                    match_list = sorted(match_list, key=lambda key: counterMgr.counter_dict[key].now_v_ratio, reverse=True)
                                    for face_index in range(for_count):
                                        # face_index = sorted_index_list[sorted_index]
                                        # result = output_results[face_index].cpu().tolist()
                                        ltrb = (bboxes[face_index][0], bboxes[face_index][1], bboxes[face_index][2], bboxes[face_index][3])
                                        score = scores[face_index]
                                        # print(f"cam_process: {ltrb}")
                                        for key in match_list:
                                            counterMgr.counter_dict[key].calFace(ltrb, score, online_im, border=border)

                else:
                    timer.toc()
                    online_im = img_info['raw_img']
                    if frame_id > int(sys.maxsize * 0.99):
                        frame_id = 0

                # if is_vis:
                counterMgr.draw_count(online_im)  # 画计数器，水印
                counterMgr.schedule(img_info['raw_img'], now)

                if args.save_result:
                    vid_writer.write(online_im)

                frame_id += 1

        except Exception as e:
            logger.info(f"{pname} {e}")
            print(f"{pname} {e}")

        if esc_event.is_set():
            if vid_writer is not None:
                vid_writer.release()
            logger.info(f"{pname} Exit!")
            print(f"{pname} Exit!")
            return


# # 启动多核进程
# def start_camera_process(pool, p_qframe, p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml):
#     # 如果可以，用配置文件重载args配置
#     if main_yaml['is_args']:
#         args.expn = cam_yaml['args']['expn']
#         args.path = cam_yaml['args']['path']
#         args.save_result = cam_yaml['args']['save_result']
#         args.exp_file = cam_yaml['args']['exp_file']
#         args.ckpt = cam_yaml['args']['ckpt']
#         args.conf = cam_yaml['args']['conf']
#         args.fps = cam_yaml['args']['fps']
#         args.track_thresh = cam_yaml['args']['track_thresh']
#         args.track_buffer = cam_yaml['args']['track_buffer']
#         args.match_thresh = cam_yaml['args']['match_thresh']
#         args.aspect_ratio_thresh = cam_yaml['args']['aspect_ratio_thresh']
#
#     parent_conn, child_conn = Pipe()  # 创建管道
#     # process_camera(output_dir, args, exp, main_yaml, cam_yaml, p_qface, p_qsql)
#     pw = pool.apply_async(func=write_process, args=(p_qframe, parent_conn, esc_event, main_yaml, cam_yaml))
#     pr = pool.apply_async(func=read_process, args=(p_qframe, child_conn, p_qface, p_qsql, esc_event,
#                                                    args, main_yaml, cam_yaml))
#
#     return pw, pr
