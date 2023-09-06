import argparse
import os
import os.path as osp
import queue
import sys
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
from tools.lzc.cam_process import Predictor


class VideoScreenshot(object):
    def __init__(self, p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml):
        self.process_name = f"[ {cam_yaml['cam_name']} Process ]"
        # print(f"{self.process_name} Launch!")

        # 外部初始化
        self.p_qface = p_qface
        self.p_qsql = p_qsql
        self.esc_event = esc_event
        self.args = args
        self.main_yaml = main_yaml
        self.cam_yaml = cam_yaml

        # 临时信息
        self.cam_url = cam_yaml['args']['path']
        # Create a Temp VideoCapture object
        cap = cv2.VideoCapture(self.cam_url)
        self.frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        self.frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        self.frame_fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"frame_width: {self.frame_width} frame_height: {self.frame_height} frame_pfs: {self.frame_fps}")
        cap.release()  # 获取相机信息后就释放掉

        # 初始化配置变量
        self.queue = Queue(cam_yaml['max_frame_size'])
        self.lzc_debug = cam_yaml['is_debug'] and main_yaml['is_debug']  # 是否打印debug信息
        self.run_mode = cam_yaml['run_mode']
        self.is_vis = cam_yaml['is_vis'] # 是否可视化线框

        # 读线程初始化
        self.output_dir = None
        self.predictor = None
        self.vid_writer = None
        self.tracker = None
        self.counterMgr = None
        self.timer = None
        self.exp_test_size = (640, 640)
        self.debug_read_frame = self.frame_fps * 5
        self._read_init() # 读先初始化，最大限度保证视频流开始时的流畅度

        # 存视频初始化
        self.update_frame = False # 每次有新帧时为True，用于写视频
        self.frame = None # 最新帧
        self.save_frame_count = 0
        self.debug_save_fps = 5
        self.screenshot_interval = 1.0 / self.frame_fps # Take screenshot every x seconds

        # 写视频初始化
        # Start the thread to read frames from the video stream
        self.debug_write_frame = self.frame_fps * 15 # 每x fps帧打印一次状态
        self.thread = threading.Thread(target=self._write_update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _read_init(self):
        args = self.args

        exp = get_exp(args.exp_file, args.name)

        # 如果没有取实验名，就取yolox_s
        if not args.experiment_name:
            args.experiment_name = exp.exp_name

        # 设置输出文件夹
        video_name = args.path.split('/')[-1].split('.')[0]
        output_dir = osp.join(exp.output_dir, args.expn, video_name)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        if args.trt:
            args.device = "gpu"
        # run_device = self.cam_yaml['device'] # 决定运行在哪块GPU上
        args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

        logger.info("Args: {}".format(args))

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        # 初始化模型
        model = exp.get_model().to(args.device)  # 初始化yolox模型
        # print(f"device: {next(model.parameters()).device}")
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()

        if not args.trt:
            if args.ckpt is None:
                ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.fp16:
            model = model.half()  # to FP16

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            # trt_file = osp.join(output_dir, "model_trt.pth")
            trt_file = osp.join(osp.dirname(args.ckpt), "model_trt.pth")
            assert osp.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/face.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        # yolox模型装饰器
        self.predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
        # current_time = time.localtime()

        save_path = osp.join(output_dir, f"{osp.basename(output_dir)}.mp4")

        if args.save_result:
            logger.info(f"video save_path is {save_path}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # fourcc = cv2.VideoWriter_fourcc(*"avc1")
            # fourcc = cv2.VideoWriter_fourcc(*'h264')
            self.vid_writer = cv2.VideoWriter(
                save_path, fourcc, self.frame_fps, (int(self.frame_width), int(self.frame_height))
            )

        self.tracker = BYTETracker(args, frame_rate=args.fps)
        self.counterMgr = create_counterMgr(self.main_yaml, self.cam_yaml, self.output_dir, self.p_qface, self.p_qsql)
        self.timer = Timer()
        self.exp_test_size = exp.test_size
        logger.info(f"{self.process_name} Init successfully from {self.cam_yaml['args']['path']}")
    def _write_update(self):
        logger.info(f"{self.process_name} Write thread work!")

        real_frame = 0
        capture = cv2.VideoCapture(self.cam_url)
        jump_thresh = 15
        now_jump = 0

        # Read the next frame from the stream in a different thread
        while True:
            # time.sleep(1.0 / self.frame_fps)
            if capture.isOpened():
                if self.queue.full(): # 队列满，被动丢帧
                    if capture.grab():
                        real_frame += 1
                else:
                    # 自适应主动丢帧
                    now_jump = int(self.queue.qsize() / jump_thresh)
                    for i in range(now_jump):
                        if capture.grab():
                            real_frame += 1

                    status, frame = capture.read()
                    if status:
                        self.queue.put(frame)
                        real_frame += 1

                if real_frame % self.debug_write_frame == 0 and self.lzc_debug:
                    logger.info(f"{self.process_name} real frame num: {real_frame}. queue.len: {self.queue.qsize()}")
            else:
                logger.info(f"{self.process_name} video stream closed, Exit!")
                print(f"{self.process_name} video stream closed, Exit!")
                break

    def update(self):
        frame_id = 0
        while True:
            if not self.queue.empty():
                frame = self.queue.get()
                now = time.time()

                if frame_id % self.debug_read_frame == 0 and self.lzc_debug:
                    logger.info(
                        f"{self.process_name} : Processing frame {frame_id} with {1. / max(1e-5, self.timer.average_time):.2f} fps")
                    self.timer.clear()

                outputs, img_info = self.predictor.inference(frame, self.timer)

                if outputs[0] is not None:
                    # if self.run_mode == 0:  # 客流模式需要特殊处理: 只保留人的boundingbox
                    outputs[0] = outputs[0][outputs[0][:, 6] == 0, :] # 都先追踪人，保留boundingbox

                    online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp_test_size)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                    self.timer.toc()

                    # 画监测区域
                    online_im = self.counterMgr.update(img_info['raw_img'], online_tlwhs, online_ids, online_scores, now,
                                                       is_draw=self.is_vis)

                    if self.is_vis:
                        online_im = plot_tracking(
                            online_im, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / self.timer.average_time
                        ) # 画追踪boundingbox

                else:
                    self.timer.toc()
                    online_im = img_info['raw_img']

                if self.is_vis:
                    self.counterMgr.draw_count(online_im) # 画计数器

                self.counterMgr.schedule(img_info['raw_img'], now)
                self.frame = online_im
                self.update_frame = True
                # if self.args.save_result and self.lzc_debug:
                #     # cv2.imshow('Video', online_im)
                #     self.vid_writer.write(online_im)

                frame_id += 1

            if self.esc_event.is_set():
                if self.vid_writer is not None:
                    self.vid_writer.release()
                logger.info(f"{self.process_name} Exit read proccess!")
                print(f"{self.process_name} Exit read proccess!")
                return

    def save_frame(self):
        # Save obtained frame periodically
        self.save_frame_count = 0
        def save_frame_thread():
            logger.info(f"{self.process_name} Save thread work!")
            while True:
                try:
                    if self.update_frame and self.vid_writer.isOpened():
                        if self.save_frame_count % (self.frame_fps * self.debug_save_fps) == 0:
                            logger.info(f"{self.process_name} save frame count: {self.save_frame_count + 1}")

                        self.vid_writer.write(self.frame)
                        self.save_frame_count += 1
                        self.update_frame = False

                    time.sleep(self.screenshot_interval)
                except AttributeError:
                    pass

        save_thread = threading.Thread(target=save_frame_thread, args=())
        save_thread.daemon = True
        save_thread.start()

# 单核多线程
def thread_process(p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml):
    pname = f"[ {cam_yaml['cam_name']} Process ]"
    # 获取当前进程的ID
    pid = os.getpid()
    logger.info(f'{pname} launch! pid: {pid}')

    video_stream_widget = VideoScreenshot(p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml)
    if args.save_result:
        video_stream_widget.save_frame()
    video_stream_widget.update()
    print(f'{pname} Exit! pid: {pid}')
    logger.info(f'{pname} Exit! pid: {pid}')

# 启动单核多线程进程
def start_thread_process(pool, p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml):
    # 如果可以，用配置文件重载args配置
    if main_yaml['is_args']:
        args.expn = cam_yaml['args']['expn']
        args.path = cam_yaml['args']['path']
        args.save_result = cam_yaml['args']['save_result']
        args.exp_file = cam_yaml['args']['exp_file']
        args.ckpt = cam_yaml['args']['ckpt']
        args.conf = cam_yaml['args']['conf']
        args.nms = cam_yaml['args']['nms']
        args.fps = cam_yaml['args']['fps']
        args.track_thresh = cam_yaml['args']['track_thresh']
        args.track_buffer = cam_yaml['args']['track_buffer']
        args.match_thresh = cam_yaml['args']['match_thresh']
        args.aspect_ratio_thresh = cam_yaml['args']['aspect_ratio_thresh']

    # thread_process(p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml)
    pool.apply_async(func=thread_process, args=(p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml))

    # return thread


# info_dict = {}
# def write_thread(queue, esc_event, main_yaml, cam_yaml):
#     global info_dict
#     fps = info_dict['cam_fps']
#     while not info_dict['read']:
#         time.sleep(1.0 / fps)
#
#     thread_name = f"[ {cam_yaml['cam_name']} Write Thread ]"
#     print(f"{thread_name}: Launch!")
#
#     cam_url = cam_yaml['args']['path']
#     real_frame = 0
#     jump_thresh = fps
#
#     cap = cv2.VideoCapture(cam_url)
#     now_jump = 0
#     lzc_debug = cam_yaml['is_debug'] and main_yaml['is_debug']  # 是否打印debug信息
#
#     while True:
#         # time.sleep(1.0 / (fps * 2 + 5))  # 延迟
#         # 退出逻辑
#         if esc_event.is_set():
#             print(f"{thread_name} Exit write thread!")
#             return
#
#         # 跳帧
#         now_jump = int(queue.qsize() / jump_thresh)
#         for i in range(now_jump):
#             _ = cap.grab()
#
#         _, img = cap.read()
#
#         if _:
#             real_frame += 1
#             # 对获取的视频帧分辨率重处理
#             # img_new = img_resize(img, width_new, height_new)
#
#             if queue.full():
#                 queue.get() # 删除最旧帧
#                 queue.put(img)  # 把img存入队列
#             else:
#                 queue.put(img)  # 把img存入队列
#         else:
#             pass
#
#         if real_frame % (fps * 5) == 0 and lzc_debug:
#             print(f"{thread_name} real frame num is {real_frame}. jump: {now_jump}")
#
#
# def read_thread(queue, p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml):
#     global info_dict
#     thread_name = f"[ {cam_yaml['cam_name']} Read Thread ]"
#     print(f"{thread_name} Launch!")
#
#     exp = get_exp(args.exp_file, args.name)
#
#     # 如果没有取实验名，就取yolox_s
#     if not args.experiment_name:
#         args.experiment_name = exp.exp_name
#
#     # 设置输出文件夹
#     video_name = args.path.split('/')[-1].split('.')[0]
#     output_dir = osp.join(exp.output_dir, args.expn, video_name)
#     os.makedirs(output_dir, exist_ok=True)
#
#     # if args.save_result:
#     #     vis_folder = osp.join(output_dir, "track_vis")
#     #     os.makedirs(vis_folder, exist_ok=True)
#
#     if args.trt:
#         args.device = "gpu"
#     args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
#
#     logger.info("Args: {}".format(args))
#
#     if args.conf is not None:
#         exp.test_conf = args.conf
#     if args.nms is not None:
#         exp.nmsthre = args.nms
#     if args.tsize is not None:
#         exp.test_size = (args.tsize, args.tsize)
#
#     # 初始化模型
#     model = exp.get_model().to(args.device)  # 初始化yolox模型
#     logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
#     model.eval()
#
#     if not args.trt:
#         if args.ckpt is None:
#             ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
#         else:
#             ckpt_file = args.ckpt
#         logger.info("loading checkpoint")
#         ckpt = torch.load(ckpt_file, map_location="cpu")
#         # load the model state dict
#         model.load_state_dict(ckpt["model"])
#         logger.info("loaded checkpoint done.")
#
#     if args.fuse:
#         logger.info("\tFusing model...")
#         model = fuse_model(model)
#
#     if args.fp16:
#         model = model.half()  # to FP16
#
#     if args.trt:
#         assert not args.fuse, "TensorRT model is not support model fusing!"
#         # trt_file = osp.join(output_dir, "model_trt.pth")
#         trt_file = osp.join(osp.dirname(args.ckpt), "model_trt.pth")
#         assert osp.exists(
#             trt_file
#         ), "TensorRT model is not found!\n Run python3 tools/face.py first!"
#         model.head.decode_in_inference = False
#         decoder = model.head.decode_outputs
#         logger.info("Using TensorRT to inference")
#     else:
#         trt_file = None
#         decoder = None
#
#     # yolox模型装饰器
#     predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
#     # current_time = time.localtime()
#
#     save_path = osp.join(output_dir, f"{osp.basename(output_dir)}.mp4")
#
#     if args.save_result:
#         cam_fps = info_dict['cam_fps']
#         cam_width = info_dict['cam_width']
#         cam_height = info_dict['cam_height']
#         logger.info(f"video save_path is {save_path}")
#         # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         vid_writer = cv2.VideoWriter(
#             save_path, fourcc, cam_fps, (int(cam_width), int(cam_height))
#         )
#     tracker = BYTETracker(args, frame_rate=args.fps)
#     timer = Timer()
#     frame_id = 0
#     results = []
#
#     # 自定义
#     lzc_debug = cam_yaml['is_debug'] and main_yaml['is_debug']  # 是否打印debug信息
#     run_mode = cam_yaml['run_mode']
#     counterMgr = create_counterMgr(main_yaml, cam_yaml, output_dir, p_qface, p_qsql)
#
#     print(f"{thread_name} Begin read from {cam_yaml['args']['path']}")
#     info_dict['read'] = True
#
#     # update部分
#     while True:
#         if not queue.empty():
#             frame = queue.get()
#             now = time.time()
#
#             if frame_id % 20 == 0 and lzc_debug:
#                 logger.info(
#                     f"{thread_name} : Processing frame {frame_id} with {1. / max(1e-5, timer.average_time):.2f} fps")
#                 timer.clear()
#
#             outputs, img_info = predictor.inference(frame, timer)
#
#             if outputs[0] is not None:
#                 if run_mode == 0:  # 客流模式需要特殊处理: 只保留人的boundingbox
#                     outputs[0] = outputs[0][outputs[0][:, 6] == 0, :]
#
#                 online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
#                 online_tlwhs = []
#                 online_ids = []
#                 online_scores = []
#                 for t in online_targets:
#                     tlwh = t.tlwh
#                     tid = t.track_id
#                     vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
#                     if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
#                         online_tlwhs.append(tlwh)
#                         online_ids.append(tid)
#                         online_scores.append(t.score)
#                         results.append(
#                             f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
#                         )
#                 timer.toc()
#
#                 img = counterMgr.update(img_info['raw_img'], online_tlwhs, online_ids, online_scores, now)
#
#                 online_im = plot_tracking(
#                     img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
#                 )
#
#             else:
#                 timer.toc()
#                 online_im = img_info['raw_img']
#
#             counterMgr.draw_count(online_im)
#             counterMgr.schedule(img_info['raw_img'], now)
#
#             if args.save_result and lzc_debug:
#                 # cv2.imshow('Video', online_im)
#                 vid_writer.write(online_im)
#
#             frame_id += 1
#
#         if esc_event.is_set():
#             vid_writer.release()
#             print(f"{thread_name} : Exit read proccess!")
#             return
