import argparse
import os
import os.path as osp
import queue
import random
import threading
import time
from multiprocessing import current_process, Pipe, Manager

import cv2
import torch
import yaml

from loguru import logger
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from tools.lzc.counter_mgr import MyCounterMgr, create_counterMgr
from queue import Queue
from tools.lzc.Predictor import create_predictor, Predictor

# 单核单线程
# 一个进程完成所有读写视频流操作
def single_process(p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml):
    pid = os.getpid()
    pname = f"[ {cam_yaml['cam_name']} Single process ]"
    print(f'{pname} launch! pid: {pid}')
    # 准备所需变量
    run_mode = cam_yaml['run_mode']

    exp = get_exp(args.exp_file, args.name)
    # 如果没有取实验名，就取yolox_s
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # 设置输出文件夹
    video_name = args.path.split('/')[-1].split('.')[0]
    output_dir = osp.join(exp.output_dir, args.expn, video_name)
    os.makedirs(output_dir, exist_ok=True)

    predictor = create_predictor(args, exp)

    # 人脸检测器
    face_predictor = None
    exp2 = None
    if main_yaml['is_args'] and run_mode == 1:
        args.exp_file = cam_yaml['args2']['exp_file']
        args.ckpt = cam_yaml['args2']['ckpt']
        args.conf = cam_yaml['args2']['conf']
        args.nms = cam_yaml['args2']['nms']
        exp2 = get_exp(args.exp_file, args.name)
        face_predictor = create_predictor(args, exp2)

    # 追踪器初始化
    tracker = BYTETracker(args, frame_rate=args.fps)

    current_time = time.localtime()
    # 获取视频流及其参数
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    # output_dir: YOLOX_outputs/renlian/test01/
    # save_path: YOLOX_outputs/renlian/test01/test01.mp4
    save_path = osp.join(output_dir, f"{osp.basename(output_dir)}.mp4")

    if args.save_result:
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    timer = Timer()
    frame_id = 0
    results = []
    match_list = []

    # 自定义
    lzc_debug = cam_yaml['is_debug'] and main_yaml['is_debug']  # 是否打印debug信息
    is_vis = cam_yaml['is_vis']
    border = cam_yaml['border']
    test_size = None if exp2 is None else exp2.test_size
    counterMgr = create_counterMgr(main_yaml, cam_yaml, output_dir, p_qface, p_qsql,
                                   face_predictor=face_predictor, test_size=test_size)
    face_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    print("初始化成功，开始测试!")
    while True:
        if frame_id % 20 == 0 and lzc_debug:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            timer.clear()

        ret_val, frame = cap.read()
        if frame_id % 2 == 0:
            frame_id += 1
            continue
        if ret_val:
            now = time.time()
            outputs, img_info = predictor.inference(frame, timer)

            if outputs[0] is not None:
                # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                # if run_mode == 0:  # 客流模式需要特殊处理: 只保留人的boundingbox
                outputs[0] = outputs[0][outputs[0][:, 6] == 0, :] # 只保留类别为人的

                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
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
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()

                counterMgr.schedule_before(now, frame_id)
                # counterMgr.can_detect_face = False
                online_im = counterMgr.update(img_info['raw_img'], online_tlwhs, online_ids, online_scores, now, frame_id)

                if is_vis:
                    online_im = plot_tracking(
                        online_im, online_tlwhs, online_ids, scores=online_scores, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )

                    # 目标检测(检测人脸)，根据脸在人内的先验，匹配人脸
                    # if run_mode == 1:
                    #     if len(counterMgr.counter_dict) > 0:  # 存在检测对象，检测人脸
                    #         # if counterMgr.can_detect_face:
                    #         face_outputs, face_img_info = face_predictor.inference(frame, timer)
                    #         if face_outputs[0] is not None:
                    #             output_results = face_outputs[0]
                    #             bboxes = output_results[:, :4]  # x1y1x2y2
                    #             scale = min(exp.test_size[0] / float(face_img_info['height']),
                    #                         exp.test_size[1] / float(face_img_info['width']))
                    #             bboxes /= scale
                    #             scores = output_results[:, 4] * output_results[:, 5]
                    #
                    #             bboxes = bboxes.cpu().tolist()
                    #             scores = scores.cpu().tolist()
                    #             for_count = face_outputs[0].shape[0]  # 根据bboxes的y轴坐标做排序
                    #             # print(f"for_count: {for_count}")
                    #             match_list.clear()
                    #             for key in counterMgr.counter_dict.keys():
                    #                 if counterMgr.counter_dict[key].get_dying() == 0 and counterMgr.counter_dict[key].get_det_flag() == 0:  # 处于运行态的才更新
                    #                     match_list.append(key)
                    #
                    #             # match_list.sort() # 根据y轴做人的排序
                    #             for face_index in range(for_count):
                    #                 # result = output_results[face_index].cpu().tolist()
                    #                 ltrb = (bboxes[face_index][0], bboxes[face_index][1], bboxes[face_index][2],
                    #                         bboxes[face_index][3])
                    #                 score = scores[face_index]
                    #                 # print(f"cam_process: {ltrb}")
                    #                 for key in match_list:
                    #                     counterMgr.counter_dict[key].calFace(ltrb, score, online_im, border=border)
            else:
                timer.toc()
                online_im = img_info['raw_img']

            online_im = counterMgr.draw_zone(online_im)
            counterMgr.draw_count(online_im)
            counterMgr.schedule(img_info['raw_img'], now)

            if args.save_result:
                vid_writer.write(online_im)

        else:
            break
        if frame_id == 100:
            cv2.imwrite(f"{os.path.join(output_dir, cam_yaml['cam_name'])}.jpg", online_im)

        if esc_event.is_set():
            break
        frame_id += 1

# 启动单核单线程进程
def start_single_process(pool, p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml):
    # 如果可以，用配置文件重载args配置
    if main_yaml['is_args']:
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

    # process_camera(output_dir, args, exp, main_yaml, cam_yaml, p_qface, p_qsql)
    p = pool.apply_async(func=single_process, args=(p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml))

    return p
