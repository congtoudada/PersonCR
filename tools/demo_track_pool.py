import argparse
import multiprocessing
import os
import sys
import threading

import termios
import tty
import time
import signal

import cv2
from loguru import logger
from tools.lzc.yaml_helper import read_yaml
from tools.lzc.cam_process import *
from tools.lzc.sql_process import *
from tools.lzc.face_process1 import *
from tools.lzc.cam_process_pool import *
from multiprocessing import Process, Manager, Pipe, current_process
# from pynput import keyboard

# IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--face",
        dest="face",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # ----------------------------- 自定义参数 -----------------------------
    parser.add_argument("--main_yaml_path", type=str, default="exps/custom/main_process1.yaml", help="main yaml path")
    return parser

# 获取键盘输入
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def key_linster(esc_event):
    # 监听键盘输入
    while True:
        key = get_key()
        if key == 'q':  # 如果按下'q'键则退出循环
            esc_event.set()
            print("Press Q, Algorithm will be over!")
            break

# 注册退出信号处理函数
def signal_handler(esc_event):
    def handler(signum, frame):
        print(f"Received signal: {signum}. Terminating!.")
        esc_event.set()

    return handler


if __name__ == "__main__":
    # 配置相关
    # 解析args
    args = make_parser().parse_args()
    # 加载yaml配置
    main_yaml = read_yaml(file=args.main_yaml_path)
    # 生成日志信息
    logger.remove()  # 避免打印到控制台
    log_path = os.path.join("logs", "demo_track_process.py {time}.log")
    logger.add(sink=log_path, rotation="100 MB")  # 每100MB重新写
    print(f"logs will save in: {log_path}")
    begin_time_str = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
    print(f"{begin_time_str}: Algorithm Running...")

    # 进程相关
    # 设置进程开启方式
    multiprocessing.set_start_method('fork')
    # multiprocessing.set_start_method('spawn')
    # 创建事件对象
    esc_event = Manager().Event()  # 控制程序终止
    sqlEvent = Manager().Event()  # sql初始化完成通知
    faceEvent = Manager().Event()  # face初始化完成通知
    # 创建进程池（方便管理）
    keliu_count = main_yaml['keliu']
    renlian_count = main_yaml['renlian']
    face_count = main_yaml['face']['count']
    sql_count = main_yaml['mysql']['count']
    pool_count = keliu_count * 2 + renlian_count * 2 + face_count + sql_count + 2 # 一个相机对应两个进程（读视频流 和 处理视频流）
    logger.info(f"进程池数量: {pool_count} 客流:{keliu_count}*2 人脸:{renlian_count}*2 人脸识别:{face_count} 数据库:{sql_count} 备用:2")
    print(f"进程池数量: {pool_count} 客流:{keliu_count}*2 人脸:{renlian_count}*2 人脸识别:{face_count} 数据库:{sql_count} 备用:2")
    pool = multiprocessing.Pool(processes=pool_count)

    # 开启数据库进程
    sql_queue_list = []
    for i in range(sql_count):
        # time.sleep(3) # 避免并发过高
        sql_queue_list.append(Manager().Queue(main_yaml['mysql']['max_size']))
        sql_name = f"sql_process {i}"
        if i == sql_count - 1:
            pool.apply_async(func=sql_process, args=(sql_name, sql_queue_list[i], sqlEvent, esc_event, main_yaml))
        else:
            pool.apply_async(func=sql_process, args=(sql_name, sql_queue_list[i], None, esc_event, main_yaml))

    # 开启人脸进程
    if sql_count != 0:
        sqlEvent.wait()
    face_queue_list = []
    for i in range(face_count):
        # time.sleep(3) # 避免并发过高
        face_queue_list.append(Manager().Queue(main_yaml['face']['max_size']))
        face_name = f"face_process {i}"
        if i == sql_count - 1:
            pool.apply_async(func=face_process, args=(face_name, face_queue_list[i], sql_queue_list[i], faceEvent, esc_event, main_yaml))
        else:
            pool.apply_async(func=face_process, args=(face_name, face_queue_list[i], sql_queue_list[i], None, esc_event, main_yaml))

    # 开启相机进程
    if face_count != 0:
        faceEvent.wait()
    # 客流相机
    # keliu_write_list = [write_keliu1]
    # keliu_read_list = [read_keliu1]
    for i in range(keliu_count):
        cam_yaml = read_yaml(file=f"exps/custom/keliu{i+1}.yaml")
        qframe = Manager().Queue(cam_yaml['max_frame_size'])
        cam_event = Manager().Event()
        # 读视频流进程
        pool.apply_async(func=write_process, args=(qframe, cam_event, esc_event, cam_yaml))
        # 处理视频流进程
        pool.apply_async(func=read_process, args=(qframe, cam_event, face_queue_list, sql_queue_list, esc_event, args, main_yaml, cam_yaml))
        time.sleep(10) # 避免并发过高,间隔10s开一个

    # 人脸相机
    renlian_write_list = [write_renlian1]
    renlian_read_list = [read_renlian1]
    for i in range(renlian_count):
        cam_yaml = read_yaml(file=f"exps/custom/renlian{i+1}.yaml")
        qframe = Manager().Queue(cam_yaml['max_frame_size'])
        cam_event = Manager().Event()
        # 读视频流进程
        pool.apply_async(func=write_process, args=(qframe, cam_event, esc_event, cam_yaml))
        # 处理视频流进程
        # read_process(qframe, cam_event, face_queue_list, sql_queue_list, esc_event, args, main_yaml, cam_yaml)
        pool.apply_async(func=read_process, args=(qframe, cam_event, face_queue_list, sql_queue_list, esc_event, args, main_yaml, cam_yaml))
        time.sleep(10) # 避免并发过高,间隔10s开一个

    # 主进程注册终止信号事件处理函数
    # signal.signal(signal.SIGINT, signal_handler(esc_event))

    # 开启键盘监听线程
    # Linux 监听，按q退出
    threading.Thread(target=key_linster, args=(esc_event,)).start()
    # Windows 监听，按q退出
    # while True:
    #     # 等待 25 毫秒
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         esc_event.set()
    #         break

    pool.close()    # 关闭进程池，不再接受新的任务
    pool.join()     # 等待所有进程完成
    print("程序结束！")
    sys.exit(0)


