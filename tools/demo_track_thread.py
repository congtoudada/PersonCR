import argparse
import multiprocessing
import os
import sys
import threading
import signal
import termios
import tty
import time

import cv2
from loguru import logger
from tools.lzc.yaml_helper import read_yaml
from tools.lzc.cam_process import start_camera_process
from tools.lzc.cam_thread import start_thread_process, thread_process
from tools.lzc.cam_single import start_single_process
from tools.lzc.sql_process import start_sql_process
from tools.lzc.face_process1 import start_face_process
from multiprocessing import Process, Manager, Pipe, current_process
# from pynput import keyboard

# IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
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
    parser.add_argument("--main_yaml_path", type=str, default="exps/custom/main.yaml", help="main yaml path")
    parser.add_argument("--cam_yaml", type=str, default="renlian1", help="camera yaml path")
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

# 监听键盘输入
def key_linster(esc_event):
    # 监听键盘输入
    while True:
        key = get_key()
        if key == 'q':  # 如果按下'q'键则退出循环
            esc_event.set()
            print("Press Q, Algorithm will be over!")
            break

# 信号处理函数
def create_signal_handler(cam_name, esc_event):
    def handle_signal(signum, frame):
        print(f"{cam_name}接收到终止信号！")
        esc_event.set()
    return handle_signal

if __name__ == "__main__":
    # 解析args
    args = make_parser().parse_args()
    # 加载yaml配置
    main_yaml = read_yaml(file=args.main_yaml_path)
    cam_yaml_path = f"exps/custom/{args.cam_yaml}.yaml"
    cam_yaml = read_yaml(cam_yaml_path)
    run_mode = cam_yaml['run_mode']

    # 设置进程开启方式
    multiprocessing.set_start_method('fork')

    # 生成日志信息
    logger.remove()  # 避免打印到控制台
    cam_name = cam_yaml['cam_name']
    log_name = cam_name + "_{time}.log"
    begin_time_str = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
    # log_path = os.path.join("logs", cam_name, begin_time_str)
    log_path = os.path.join("logs", cam_name)
    logger.add(sink=os.path.join(log_path, log_name), rotation="100 MB") # 每100MB重新写
    mode_name = "Keliu" if cam_yaml['run_mode'] == 0 else "Renlian"
    print(f"logs will save in: {log_path}")
    print(f"{begin_time_str}: {mode_name} Algorithm Running...")

    p_qsql = Manager().Queue(main_yaml['mysql']['max_size'])
    p_qface = None
    if run_mode == 1:
        p_qface = Manager().Queue(main_yaml['face']['max_size'])
    else:
        p_qface = Manager().Queue(1)

    # 创建事件对象
    esc_event = Manager().Event()  # 控制程序终止
    sqlEvent = Manager().Event()  # sql初始化完成通知
    faceEvent = Manager().Event()  # face初始化完成通知

    pface = None
    psql = None
    # 开启数据库进程
    psql = start_sql_process(p_qsql, sqlEvent, esc_event, main_yaml)
    sqlEvent.wait()

    # 开启人脸进程
    if cam_yaml['run_mode'] == 1:
        pface = start_face_process(p_qface, p_qsql, faceEvent, esc_event, args, main_yaml)
        faceEvent.wait()


    # 开启键盘监听线程
    # threading.Thread(target=key_linster, args=(esc_event,)).start()
    # 注册终止信号
    signal_handler = create_signal_handler(cam_name, esc_event)
    signal.signal(signal.SIGINT, signal_handler)

    # 核心算法
    pool = None
    start_thread_process(pool, p_qface, p_qsql, esc_event, args, main_yaml, cam_yaml)

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    # if pface is not None:
    #     pface.terminate()
    #
    # if psql is not None:
    #     psql.terminate()
    pface.join()
    psql.join()

    print(f"{cam_name} 相关资源清理完毕")
    sys.exit(0)

