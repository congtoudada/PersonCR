import os
import pickle
import threading
import time
import traceback

from loguru import logger
from multiprocessing import Process, current_process, Manager
import cv2
import numpy as np

from tools.lzc.config_tool import ConfigTool
from tools.lzc.face.framework.FaceRegMgr import FaceRegMgr
from tools.lzc.face.framework.FaceRegTool import FaceRegTool
from tools.lzc.face.framework.IFaceRegMgr import IFaceRegMgr

"""
process_id: 自定义进程id
faceQueue: 人脸消息队列
qresult_list: 存储人脸结果的消息队列列表
faceEvent: 人脸进程初始化事件
escEvent: 退出事件
main_yaml: 主配置文件
"""
def face_process(process_id, faceReq_queue_list: list, faceRsp_queue_list: list, faceEvent, escEvent, main_yaml):
    pname = f'[ {os.getpid()}:face_process {process_id} ]'

    # 使用spawn会创建全新进程，需重新配置日志
    ConfigTool.load_log_config(main_yaml)

    sleep_time = main_yaml['face']['sleep']
    main_id = main_yaml['main_id']
    face_update_path = main_yaml['face']['update_path']
    score_thresh = main_yaml['face']['score_thresh']

    can_update = True if main_id == 1 and process_id == 1 else False
    faceRegMgr: IFaceRegMgr = FaceRegMgr(face_update_path, can_update)

    logger.info(f'{pname} launch!')

    if faceEvent is not None:
        faceEvent.set()  # 初始化结束

    try:
        while True:
            for i in range(faceReq_queue_list.__len__()):
                if not faceReq_queue_list[i].empty():
                    pack_req_data = faceReq_queue_list[i].get()

                    obj_id, img = FaceRegTool.unpack_req(pack_req_data)

                    if img is not None:
                        per_id, score = faceRegMgr.inference(img)
                        logger.info(
                            f"{pname} 人脸识别结果 [obj:{obj_id} per: {per_id} score: {score:.2f}]")
                        if per_id != 1 and score < score_thresh:
                            # logger.info(f"{pname} obj_id:{obj_id} - per_id: {per_id} 识别分数过低被剔除: {score:.2f} < {score_thresh:.2f}")
                            per_id = 1  # 设为Unknown
                    else:
                        logger.error(f"{pname} 人脸图片获取失败!")
                        continue

                    if per_id == 1:
                        continue

                    # logger.info(f"{pname} 识别成功! obj: {obj_id} match result:{per_id}")
                    pack_rsp_data = FaceRegTool.pack_rsp(obj_id, per_id, score, img)
                    faceRsp_queue_list[i].put(pack_rsp_data)

            else:  # 检测特征库更新，是否退出进程
                if faceRegMgr.check_and_update():
                    logger.info(f"{pname} 检测到特征库变化，更新人脸特征库")

                if escEvent.is_set():
                    logger.info(f"{pname} Exit!")
                    return

                time.sleep(sleep_time)  # 暂停1s，让出控制权

    except Exception as e:
        logger.error(f"{pname} Error:{traceback.format_exc()}")


if __name__ == "__main__":
    main_yaml = ConfigTool.load_main_config("exps/custom/main1.yaml")
    ConfigTool.load_log_config(main_yaml)

    main_id = main_yaml['main_id']
    face_update_path = main_yaml['face']['update_path']

    faceRegMgr: IFaceRegMgr = FaceRegMgr(face_update_path, True)
    # 使用str加载
    # per_id, score = faceRegMgr.inference("./assets/face/test/test1_4.jpg")

    # 使用ndarry加载

    # Load an image using OpenCV
    image = cv2.imread("./assets/face/test/test1_4.jpg")  # Replace 'your_image.jpg' with your image file path

    # Check if the image was loaded successfully
    if image is not None:
        # Convert the image to a NumPy array
        image_array = np.array(image)
        # Now, you can manipulate and process the image using NumPy operations
        # Display the shape of the image array
        print("Image shape:", image_array.shape)
    else:
        print("Image not found or could not be loaded.")

    per_id, score = faceRegMgr.inference(image)
    logger.info("人脸识别结果: " + str(per_id) + " 分数: " + str(score))






