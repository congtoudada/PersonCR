import os
import pickle
import threading
import time

from loguru import logger
from multiprocessing import Process, current_process

from facenet.facenet import Facenet
# from facenet.mypredict import mypredict
from tools.lzc.my_logger import log_config
from Face_Recognition import FaceModel, configs, my_register


def face_reboot(face_update_path, main_id, process_id, face_model=None):
    if main_id == 1 and process_id == 1:  # 由1号子进程的1号人脸进程生成json文件
        my_register(configs.portrait, configs.database, is_local=False)
        # 成功更新特征库，生成文件标识
        if not os.path.exists(face_update_path):
            write_data = {"face": "running"}
            pickle.dump(write_data, open(face_update_path, 'wb'))
    else:
        while not os.path.exists(configs.database):  # 其他进程等待特征库生成完毕后初始化
            time.sleep(0.5)

    while not os.path.exists(face_update_path):
        time.sleep(0.5)  # 等待json生成完毕

    if face_model is None:
        face_model = FaceModel(database=configs.database)  # 加载新的人脸识别模型
    else:
        face_model.faceReg.face_database = face_model.faceReg.load(file=configs.database)
        face_model.faceReg.update()
    return face_model


def face_process(process_id, p_qface, p_qsql, faceEvent, esc_event, main_yaml):
    pname = f'[ {os.getpid()}:face_process {process_id} ]'
    logger.info(f'{pname} launch!')
    print(f'{pname} launch!')
    log_config(main_id=main_yaml['main_id'])

    sleep_time = main_yaml['face']['sleep']
    is_debug = main_yaml['is_debug']
    per_img_count = main_yaml['face']['per_img_count']  # 一个人最多处理几张候选图
    # database = main_yaml['face']['database']
    main_id = main_yaml['main_id']
    face_update_path = main_yaml['face']['update_path']

    if is_debug:
        print(f"{pname} 算法首次启动，更新人脸特征库")
        logger.info(f"{pname} 算法首次启动，更新人脸特征库")
    face_model = face_reboot(face_update_path, main_id, process_id, face_model=None)  # 加载人脸识别模型

    if faceEvent is not None:
        faceEvent.set()  # 初始化结束

    try:
        while True:
            if not p_qface.empty():
                # print(f"face len: {len(p_qface)}")
                if not os.path.exists(face_update_path):  # 特征库存在才进行检测
                    time.sleep(sleep_time)
                    continue

                val_dict = p_qface.get()
                # 初始化一些所需变量
                # print(f"val_dict: {val_dict}")
                best_save_path = val_dict['record_photo_url']
                iter_path = best_save_path

                per_id = 1
                for i in range(per_img_count):
                    if os.path.exists(iter_path):
                        per_id, score = face_model.detect_image(iter_path)
                        if per_id != 1:  # 非陌生人直接返回
                            break
                        else:  # 否则选取下一张图片
                            iter_path = best_save_path.replace(".jpg", f"{i + 1}.jpg")
                    else:
                        break

                if is_debug:
                    logger.info(f"{pname} detect face! obj: {val_dict['obj_id']} match result:{per_id}")  # 人脸预测里面会打印日志
                    print(f"{pname} detect face! obj: {val_dict['obj_id']} match result:{per_id}")

                if val_dict['is_sql']:
                    if not p_qsql.full():
                        p_qsql.put({
                            "run_mode": val_dict['run_mode'],
                            "record_time": val_dict['record_time'],
                            "recognize_cam_id": val_dict['recognize_cam_id'],
                            "record_status": val_dict['record_status'],
                            "record_num": 1,
                            "record_photo_url": best_save_path,
                            "personnel_id": per_id,
                            "is_warning": 1 if per_id == 1 else 0,
                            "record_video_url": val_dict['record_video_url'],
                        })
                    else:
                        logger.error(f"{pname} Mysql Queue is full! Maybe MySQL is closed!")
            else:
                if not os.path.exists(face_update_path):  # 检测到特征库更新
                    if is_debug:
                        print(f"{pname} 检测到特征库变化，更新人脸特征库")
                        logger.info(f"{pname} 检测到特征库变化，更新人脸特征库")
                    face_model = face_reboot(face_update_path, main_id, process_id, face_model=face_model)  # 闲置时更新

                if esc_event.is_set():
                    logger.info(f"{pname} Exit!")
                    print(f"{pname} Exit!")
                    return

                time.sleep(sleep_time)  # 暂停1s，让出控制权
    except Exception as e:
        print(f"{pname} Error:{e}")
        logger.error(f"{pname} Error:{e}")


def start_face_process(p_qface, p_qsql, faceEvent, esc_event, args, main_yaml):
    pface = Process(target=face_process, name="face_process",
                    args=(p_qface, p_qsql, faceEvent, esc_event, args, main_yaml))
    pface.start()
    return pface
