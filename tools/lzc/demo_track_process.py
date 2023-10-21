import multiprocessing

from tools.lzc.process.cam_process import *
from tools.lzc.process.sql_process import *
from tools.lzc.process.face_process import *
from multiprocessing import Process, Manager


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
    parser.add_argument('--min_box_area', type=float, default=5, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # ----------------------------- 自定义参数 -----------------------------
    parser.add_argument("--main", type=str, default="main1", help="main yaml path")
    return parser


def run():
    # 解析args
    args = make_parser().parse_args()

    # 加载yaml主配置
    main_yaml_template = read_yaml(file=f"exps/custom/template/main_template.yaml")
    main_yaml = read_yaml(file=f"exps/custom/{args.main}.yaml")
    # 使用main_yaml值去更新main_yaml_template
    main_yaml_template.update((key, value) for key, value in main_yaml.items() if key in main_yaml_template)
    main_yaml: dict = main_yaml_template

    # 加载日志配置
    if main_yaml.get("enable_log"):
        logger.remove()  # 避免打印到控制台
        log_path = os.path.join("logs", f"main_process{main_yaml['main_id']}.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        logger.add(sink=log_path, rotation="100 MB")  # 每100MB重新写
        print(f"logs will save in: {log_path}")
        begin_time_str = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
        print(f"{begin_time_str}: Algorithm Running...")

    # 进程相关
    # 设置进程开启方式
    # multiprocessing.set_start_method('fork')
    multiprocessing.set_start_method('spawn')
    # 创建事件对象
    esc_event = Manager().Event()  # 控制程序终止
    sqlEvent = Manager().Event()  # sql初始化完成通知
    faceEvent = Manager().Event()  # face初始化完成通知
    # 创建进程池（方便管理）
    cam_list = main_yaml['cam']['list']
    cam_count = len(cam_list)
    face_count = main_yaml['face']['count']
    sql_count = main_yaml['sql']['count']
    process_count = cam_count * 2 + face_count + sql_count  # 一个相机对应两个进程（读视频流 和 处理视频流）
    logger.info(f"进程数量: {process_count} 相机:{cam_count}*2 人脸识别:{face_count} 数据库:{sql_count}")
    print(f"进程数量: {process_count} 相机:{cam_count}*2 人脸识别:{face_count} 数据库:{sql_count}")

    # 创建需要的队列
    sql_queue_list = []
    for i in range(sql_count):
        sql_queue_list.append(Manager().Queue(main_yaml['sql']['max_size']))
    face_queue_list = []
    for i in range(face_count):
        face_queue_list.append(Manager().Queue(main_yaml['face']['max_size']))
    cam_interval = main_yaml['cam_interval']  # 每5s启动一台相机
    max_frame_size = main_yaml['max_frame_size']
    qframe_list = []
    cam_event_list = []
    for i in range(cam_count):
        qframe_list.append(Manager().Queue(max_frame_size))
        cam_event_list.append(Manager().Event())

    # 开启数据库进程
    for i in range(sql_count):
        # time.sleep(3) # 避免并发过高
        sql_id = i + 1
        sql_name = f"sql_process {sql_id}"
        Process(target=sql_process,
                args=(sql_id, sql_queue_list[i], sqlEvent if i == sql_count - 1 else None, esc_event, main_yaml),
                name=sql_name,
                daemon=True).start()

    # 开启人脸进程
    if sql_count != 0:
        sqlEvent.wait()

    for i in range(face_count):
        # time.sleep(3) # 避免并发过高
        face_id = i + 1
        face_name = f"face_process {face_id}"
        Process(target=face_process,
                args=(
                    face_id, face_queue_list[i], sql_queue_list[i], faceEvent if i == face_count - 1 else None,
                    esc_event,
                    main_yaml),
                name=face_name,
                daemon=True).start()

    # 开启相机进程
    if face_count != 0:
        faceEvent.wait()

    for i in range(cam_count):
        cam_yaml = read_yaml(file=f"exps/custom/{cam_list[i]}.yaml")
        # 读视频流进程
        Process(target=write_process,
                args=(qframe_list[i], cam_event_list[i], esc_event, main_yaml, cam_yaml),
                daemon=True).start()

        # 处理视频流进程
        Process(target=read_process,
                args=(qframe_list[i], cam_event_list[i], face_queue_list, sql_queue_list, esc_event, args, main_yaml,
                      cam_yaml),
                daemon=True).start()
        time.sleep(cam_interval)  # 避免并发过高,间隔10s开一个

    # 写文件用于监听，文件删除算法终止
    main_path = "./assets/face/main/main.pkl"
    if not os.path.exists(main_path):
        write_data = {"main": "running"}
        pickle.dump(write_data, open(main_path, 'wb'))

    while True:
        time.sleep(1)
        if not os.path.exists(main_path):  # 检测系统运行
            break

    if not esc_event.is_set():
        esc_event.set()

    print("算法将在3s后终止！")
    count_down = 3
    for i in range(count_down):
        print(f"{3 - i}")
        time.sleep(1)

    # 删除人脸特征库标识文件
    if main_yaml['main_id'] == 1:
        if os.path.exists(main_yaml['face']['update_path']):
            os.remove(main_yaml['face']['update_path'])

    print("程序结束！")
    sys.exit(0)


# 日志测试
def test1_log():
    # 解析args
    args = make_parser().parse_args()

    # 加载yaml主配置
    main_yaml = ConfigTool.load_main_config(args.main)

    # 初始化日志模块
    ConfigTool.load_log_config(main_yaml, is_clean=True)

    logger.info(f"Algorithm is Running...")


def test2_init_queue():
    # ---------- 配置相关 ----------
    # 解析args
    args = make_parser().parse_args()

    # 加载yaml主配置
    main_yaml = ConfigTool.load_main_config(args.main)

    # 初始化日志模块
    ConfigTool.load_log_config(main_yaml, is_clean=True)

    logger.info(f"Algorithm is Running...")

    # ---------- 进程相关 ----------
    # 设置进程开启方式
    # multiprocessing.set_start_method('spawn')
    # 创建事件对象
    esc_event = Manager().Event()  # 程序终止事件
    sqlEvent = Manager().Event()  # 数据库初始化完成事件
    faceEvent = Manager().Event()  # 人脸识别模块初始化完成事件
    # 创建进程池（方便管理）
    cam_list = main_yaml['cam']['list']
    cam_count = len(cam_list)
    face_count = main_yaml['face']['count']
    sql_count = main_yaml['database']['count']
    process_count = cam_count * 2 + face_count + sql_count  # 一个相机对应两个进程（读视频流 和 处理视频流）
    logger.info(f"进程数量: {process_count} 相机:{cam_count}*2 人脸识别:{face_count} 数据库:{sql_count}")

    # 创建需要的消息队列
    # 数据库消息队列
    sql_queue_list = []
    for i in range(sql_count):
        sql_queue_list.append(Manager().Queue(main_yaml['database']['capacity']))
    # 人脸识别消息队列
    face_queue_list = []
    for i in range(face_count):
        face_queue_list.append(Manager().Queue(main_yaml['face']['capacity']))
    # 相机消息消息队列
    cam_interval = main_yaml['cam_interval']  # 每n秒启动一台相机
    frame_queue_capacity = main_yaml['cam']['frame_queue_capacity']  # 视频帧缓存队列上限
    qframe_list = []  # 存放读取的视频帧
    qface_list = []  # 存放人脸识别返回的结果
    cam_event_list = []  # 用于初始化读/写视频流进程
    for i in range(cam_count):
        qframe_list.append(Manager().Queue(frame_queue_capacity))
        qface_list.append(Manager().Queue(frame_queue_capacity))
        cam_event_list.append(Manager().Event())

    logger.info("init over!")


def test3_init_database():
    # ---------- 配置相关 ----------
    # 解析args
    args = make_parser().parse_args()

    # 加载yaml主配置
    main_yaml = ConfigTool.load_main_config(args.main)

    # 初始化日志模块
    ConfigTool.load_log_config(main_yaml, is_clean=True)

    logger.info(f"Algorithm is Running...")

    # ---------- 进程相关 ----------
    # 设置进程开启方式
    # multiprocessing.set_start_method('spawn')
    # 创建事件对象
    escEvent = Manager().Event()  # 程序终止事件
    sqlEvent = Manager().Event()  # 数据库初始化完成事件
    faceEvent = Manager().Event()  # 人脸识别模块初始化完成事件
    # 创建进程池（方便管理）
    cam_list = main_yaml['cam']['list']
    cam_count = len(cam_list)
    face_count = main_yaml['face']['count']
    sql_count = main_yaml['database']['count']
    process_count = cam_count * 2 + face_count + sql_count  # 一个相机对应两个进程（读视频流 和 处理视频流）
    logger.info(f"进程数量: {process_count} 相机:{cam_count}*2 人脸识别:{face_count} 数据库:{sql_count}")

    # 创建需要的消息队列
    # 数据库消息队列
    sql_queue_list = []
    for i in range(sql_count):
        sql_queue_list.append(Manager().Queue(main_yaml['database']['capacity']))
    # 人脸识别消息队列
    face_queue_list = []
    for i in range(face_count):
        face_queue_list.append(Manager().Queue(main_yaml['face']['capacity']))
    # 相机消息消息队列
    cam_interval = main_yaml['cam_interval']  # 每n秒启动一台相机
    frame_queue_capacity = main_yaml['cam']['frame_queue_capacity']  # 视频帧缓存队列上限
    qframe_list = []  # 存放读取的视频帧
    qresult_list = []  # 存放人脸识别返回的结果
    cam_event_list = []  # 用于初始化读/写视频流进程
    for i in range(cam_count):
        qframe_list.append(Manager().Queue(frame_queue_capacity))
        qresult_list.append(Manager().Queue(frame_queue_capacity))
        cam_event_list.append(Manager().Event())

    # 开启数据库进程
    if sql_count > 0:
        for i in range(sql_count):
            sql_id = i + 1
            sql_name = f"sql_process {sql_id}"
            Process(target=sql_process,
                    args=(sql_id, sql_queue_list[i],
                          sqlEvent if i == sql_count - 1 else None,
                          escEvent, main_yaml),
                    name=sql_name,
                    daemon=True).start()
    else:
        sqlEvent.set()

    sqlEvent.wait()
    logger.info("数据库初始化完成！")

    time.sleep(10)
    logger.info("程序结束！")


def test4_init_face():
    # ---------- 配置相关 ----------
    # 解析args
    args = make_parser().parse_args()

    # 加载yaml主配置
    main_yaml = ConfigTool.load_main_config(args.main)

    # 初始化日志模块
    ConfigTool.load_log_config(main_yaml, is_clean=True)

    logger.info(f"Algorithm is Running...")

    # ---------- 进程相关 ----------
    # 设置进程开启方式
    # multiprocessing.set_start_method('spawn')
    # 创建事件对象
    escEvent = Manager().Event()  # 程序终止事件
    sqlEvent = Manager().Event()  # 数据库初始化完成事件
    faceEvent = Manager().Event()  # 人脸识别模块初始化完成事件
    # 创建进程池（方便管理）
    cam_list = main_yaml['cam']['list']
    cam_count = len(cam_list)
    face_count = main_yaml['face']['count']
    sql_count = main_yaml['database']['count']
    process_count = cam_count * 2 + face_count + sql_count  # 一个相机对应两个进程（读视频流 和 处理视频流）
    logger.info(f"进程数量: {process_count} 相机:{cam_count}*2 人脸识别:{face_count} 数据库:{sql_count}")

    # 创建需要的消息队列
    # 数据库消息队列
    sql_queue_list = []
    for i in range(sql_count):
        sql_queue_list.append(Manager().Queue(main_yaml['database']['capacity']))

    # 人脸识别消息队列
    faceReq_queue_list = []
    faceRsp_queue_list = []
    faceReq_queue_list_list = []
    faceRsp_queue_list_list = []
    faceReg_capacity = main_yaml['face']['capacity']
    for i in range(face_count):
        for j in range(int(cam_count / face_count)):
            faceReq_queue_list.append(Manager().Queue(faceReg_capacity))
            faceRsp_queue_list.append(Manager().Queue(faceReg_capacity))
        faceReq_queue_list_list.append(faceReq_queue_list)
        faceRsp_queue_list_list.append(faceRsp_queue_list)
        faceReq_queue_list.clear()
        faceRsp_queue_list.clear()

    rest_count = cam_count % face_count
    if rest_count != 0:
        for i in range(rest_count):
            faceReq_queue_list_list[i].append(Manager().Queue(faceReg_capacity))
            faceRsp_queue_list_list[i].append(Manager().Queue(faceReg_capacity))

    # 相机消息消息队列
    cam_interval = main_yaml['cam_interval']  # 每n秒启动一台相机
    frame_queue_capacity = main_yaml['cam']['frame_queue_capacity']  # 视频帧缓存队列上限
    cam_event_list = []  # 用于初始化读/写视频流进程
    for i in range(cam_count):
        cam_event_list.append(Manager().Event())

    # PS:启动进程无法传对象，No OOP
    # 开启数据库进程
    if sql_count > 0:
        for i in range(sql_count):
            sql_id = i + 1
            sql_name = f"sql_process {sql_id}"
            Process(target=sql_process,
                    args=(sql_id, sql_queue_list[i],
                          sqlEvent if i == sql_count - 1 else None,
                          escEvent, main_yaml),
                    name=sql_name,
                    daemon=True).start()
    else:
        sqlEvent.set()

    sqlEvent.wait()
    logger.info("数据库模块加载完成！")

    if face_count > 0:
        for i in range(face_count):
            face_id = i + 1
            face_name = f"face_process {face_id}"
            Process(target=face_process,
                    args=(
                        face_id, faceReq_queue_list_list[i], faceRsp_queue_list_list[i],
                        faceEvent if i == face_count - 1 else None,
                        escEvent, main_yaml),
                    name=face_name,
                    daemon=True).start()
    else:
        faceEvent.set()

    faceEvent.wait()
    logger.info("人脸识别模块加载完成！")


def test5_init_all():
    # ---------- 配置相关 ----------
    # 解析args
    args = make_parser().parse_args()

    # 加载yaml主配置

    main_yaml = ConfigTool.load_main_config(f"exps/custom/{args.main}.yaml")

    # 初始化日志模块
    ConfigTool.load_log_config(main_yaml, is_clean=True)

    logger.info(f"Algorithm is Running...")

    # ---------- 子进程相关 ----------
    # 设置进程开启方式
    # multiprocessing.set_start_method('spawn')
    # 创建事件对象
    escEvent = Manager().Event()  # 程序终止事件
    sqlEvent = Manager().Event()  # 数据库初始化完成事件
    faceEvent = Manager().Event()  # 人脸识别模块初始化完成事件
    # 创建进程池（方便管理）
    cam_list = main_yaml['cam']['list']
    cam_count = len(cam_list)
    face_count = main_yaml['face']['count']
    sql_count = main_yaml['database']['count']
    process_count = cam_count * 2 + face_count + sql_count  # 一个相机对应两个进程（读视频流 和 处理视频流）
    logger.info(f"进程数量: {process_count} 相机:{cam_count}*2 人脸识别:{face_count} 数据库:{sql_count}")

    # 创建需要的消息队列
    # 数据库消息队列
    sql_queue_list = []
    for i in range(sql_count):
        sql_queue_list.append(Manager().Queue(main_yaml['database']['capacity']))

    # 人脸识别消息队列
    faceReq_queue_list = []
    faceRsp_queue_list = []
    face_capacity = main_yaml['face']['capacity']
    for i in range(cam_count):
        faceReq_queue_list.append(Manager().Queue(face_capacity))
        faceRsp_queue_list.append(Manager().Queue(face_capacity))

    # 相机消息消息队列
    cam_interval = main_yaml['cam_interval']  # 每n秒启动一台相机
    frame_queue_capacity = main_yaml['cam']['frame_queue_capacity']  # 视频帧缓存队列上限
    cam_event_list = []  # 用于初始化读/写视频流进程
    qframe_list = []
    for i in range(cam_count):
        cam_event_list.append(Manager().Event())
        qframe_list.append(Manager().Queue(frame_queue_capacity))

    # PS:启动子进程无法传对象，No OOP
    # 开启数据库进程
    if sql_count > 0:
        for i in range(sql_count):
            sql_id = i + 1
            sql_name = f"sql_process {sql_id}"
            Process(target=sql_process,
                    args=(sql_id, sql_queue_list[i],
                          sqlEvent if i == sql_count - 1 else None,
                          escEvent, main_yaml),
                    name=sql_name,
                    daemon=True).start()
    else:
        sqlEvent.set()

    sqlEvent.wait()
    logger.info("数据库模块加载完成！")

    if face_count > 0:
        avg = int(cam_count / face_count)
        mod = int(cam_count % face_count)
        left = 0
        right = 0

        for i in range(face_count):
            left = avg * i
            right = avg * i + avg
            if i == face_count - 1:
                right += mod
            face_id = i + 1
            face_name = f"face_process {face_id}"
            Process(target=face_process,
                    args=(
                        face_id, faceReq_queue_list[left: right], faceRsp_queue_list[left: right],
                        faceEvent if i == face_count - 1 else None,
                        escEvent, main_yaml),
                    name=face_name,
                    daemon=True).start()
    else:
        faceEvent.set()

    faceEvent.wait()
    logger.info("人脸识别模块加载完成！")

    for i in range(cam_count):
        cam_yaml = ConfigTool.load_cam_config(cam_file=f"exps/custom/{cam_list[i]}.yaml")
        # 读视频流进程
        Process(target=write_process,
                args=(qframe_list[i], cam_event_list[i], escEvent, main_yaml, cam_yaml),
                daemon=True).start()

        # 处理视频流进程
        Process(target=read_process,
                args=(
                    qframe_list[i], cam_event_list[i], faceReq_queue_list[i], faceRsp_queue_list[i],
                    sql_queue_list, escEvent, args, main_yaml, cam_yaml),
                daemon=True).start()
        time.sleep(cam_interval)  # 避免并发过高,间隔cam_interval秒开一个

    # ---------- 主进程相关 ----------
    # 写文件用于监听，文件删除算法终止
    main_path = main_yaml['running_file']
    if not os.path.exists(main_path):
        write_data = {"main": "running"}
        pickle.dump(write_data, open(main_path, 'wb'))

    while True:
        time.sleep(1)
        if not os.path.exists(main_path): # 检测系统运行
            break

    if not escEvent.is_set():
        escEvent.set()

    logger.info("算法将在3s后终止！")
    count_down = 3
    for i in range(count_down):
        logger.info(f"倒计时: {3 - i}")
        time.sleep(1)

    # 删除人脸特征库标识文件
    if main_yaml['main_id'] == 1:
        if os.path.exists(main_yaml['face']['update_path']):
            os.remove(main_yaml['face']['update_path'])

    logger.info("程序结束！")
    sys.exit(0)

if __name__ == "__main__":
    # test1_log()
    # test2_init_queue()
    # test3_init_database()
    test5_init_all()
