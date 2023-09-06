# import os
# import pickle
# import threading
# import time
#
# from loguru import logger
# from multiprocessing import Process, current_process
#
# from facenet.facenet import Facenet
# # from facenet.mypredict import mypredict
# # from Face_Recognition import register, FaceModel, configs
# from tools.lzc.face_helper import FaceHelper
#
#
# def face_reboot(face_update_path, main_id):
#     face_model = None
#     if main_id == 1:
#         # register(is_load=False)  # 更新人脸特征库，不从缓存中加载
#         # face_model = FaceModel(database=face_update_path)  # 加载人脸识别模型
#         face_model = FaceHelper(is_force_update=True)
#         # 成功更新特征库，生成文件标识
#         if not os.path.exists(face_update_path):
#             write_data = {"face": "running"}
#             pickle.dump(write_data, open(face_update_path, 'wb'))
#     else:
#         while True:
#             if os.path.exists(face_update_path):
#                 face_model = FaceHelper(is_force_update=False)
#                 # face_model = FaceModel(database=face_update_path)  # 加载人脸识别模型
#                 time.sleep(0.01)
#                 break
#
#     return face_model
#
# def face_process(process_name, p_qface, p_qsql, faceEvent, esc_event, main_yaml):
#     pname = f'[ {os.getpid()}:{process_name} ]'
#     logger.info(f'{pname} launch!')
#     print(f'{pname} launch!')
#
#     sleep_time = main_yaml['face']['sleep']
#     # database = main_yaml['face']['database']
#     main_id = main_yaml['main_id']
#     face_update_path = main_yaml['face']['update_path']
#
#
#     if main_id == 1:
#         print(f"{pname} 算法首次启动，更新人脸特征库")
#         logger.info(f"{pname} 算法首次启动，更新人脸特征库")
#
#     # facehelper = FaceHelper(is_force_update=True)
#     face_model = face_reboot(face_update_path, main_id)  # 加载人脸识别模型
#
#     if faceEvent is not None:
#         faceEvent.set()  # 初始化结束
#
#     try:
#
#         while True:
#             if not p_qface.empty():
#                 # print(f"face len: {len(p_qface)}")
#                 if not os.path.exists(face_update_path): # 特征库存在才进行检测
#                     time.sleep(sleep_time)
#                     continue
#
#                 val_dict = p_qface.get()
#                 # print(f"val_dict: {val_dict}")
#                 best_save_path = val_dict['record_photo_url']
#                 # name, prob = mypredict(best_save_path, face_model, db_path=face_db_path)
#                 if os.path.exists(best_save_path):
#                     # per_id = face_model.detect_image(best_save_path)
#                     per_id = face_model.predict(best_save_path)
#                 else:  # 没抓拍到人脸
#                     per_id = 1
#
#                 # logger.info(f"{pname}:{pid} detect face! obj: {val_dict['obj_id']} match result:{per_id}") # 人脸预测里面会打印日志
#                 print(f"{pname} detect face! obj: {val_dict['obj_id']} match result:{per_id}")
#
#                 if val_dict['is_sql']:
#                     if not p_qsql.full():
#                         p_qsql.put({
#                             "run_mode": val_dict['run_mode'],
#                             "record_time": val_dict['record_time'],
#                             "recognize_cam_id": val_dict['recognize_cam_id'],
#                             "record_status": val_dict['record_status'],
#                             "record_num": 1,
#                             "record_photo_url": best_save_path,
#                             "personnel_id": per_id,
#                             "is_warning": 1 if per_id == 1 else 0,
#                             "record_video_url": val_dict['record_video_url'],
#                         })
#                     else:
#                         logger.error(f"{pname} Mysql Queue is full! Maybe MySQL is closed!")
#             else:
#                 if not os.path.exists(face_update_path):  # 检测到特征库更新
#                     print(f"{pname} 检测到特征库变化，更新人脸特征库")
#                     logger.info(f"{pname} 检测到特征库变化，更新人脸特征库")
#                     face_model = face_reboot(face_update_path, main_id) # 闲置时更新
#
#
#                 if esc_event.is_set():
#                     logger.info(f"{pname} Exit!")
#                     print(f"{pname} Exit!")
#                     return
#
#                 time.sleep(sleep_time)  # 暂停1s，让出控制权
#     except Exception as e:
#         print(f"{pname} Error:{e}")
#         logger.error(f"{pname} Error:{e}")
#
# def start_face_process(p_qface, p_qsql, faceEvent, esc_event, args, main_yaml):
#     pface = Process(target=face_process, name="face_process",
#                     args=(p_qface, p_qsql, faceEvent, esc_event, args, main_yaml))
#     pface.start()
#     return pface
