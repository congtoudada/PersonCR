# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-04-05 14:09:34
# --------------------------------------------------------
"""
import os
import cv2
import argparse
import traceback
from Face_Recognition.configs import configs
from Face_Recognition.core import face_recognizer
from pybaseutils import image_utils, file_utils


class FaceModel(face_recognizer.FaceRecognizer):
    def __init__(self, database):
        """
        @param database: 人脸数据库的路径
        """
        super(FaceModel, self).__init__(database=database)

    def start_capture(self, video_file, save_video=None, detect_freq=1, vis=True):
        """
        start capture video
        :param video_file: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        video_cap = image_utils.get_video_capture(video_file)
        width, height, numFrames, fps = image_utils.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_utils.get_video_writer(save_video, width, height, fps)
        count = 0
        while True:
            if count % detect_freq == 0:
                # 设置抽帧的位置
                if isinstance(video_file, str): video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                isSuccess, frame = video_cap.read()
                if not isSuccess:
                    break
                frame, face_info = self.search_face_task(frame, thickness=4, fontScale=2.0, delay=10, vis=True)
                if save_video:
                    self.video_writer.write(frame)
            count += 1
        video_cap.release()

    def detect_image_dir(self, image_dir, out_dir=None, vis=True):
        """
        @param image_dir:
        @param out_dir:
        @param vis:
        @return:
        """
        image_list = file_utils.get_files_lists(image_dir, postfix=file_utils.IMG_POSTFIX)
        for image_file in image_list:
            try:
                print("image_file:{}\t".format(image_file), end=',', flush=True)
                image = image_utils.read_image_ch(image_file)
                image = image_utils.resize_image(image, size=(None, 640))
                image, face_info = self.search_face_task(image, vis=vis)
                print(f"id:{face_info['label'][0]} score:{face_info['score'][0]}")
                if out_dir:
                    out_file = file_utils.create_dir(out_dir, None, os.path.basename(image_file))
                    print("save result:{}".format(out_file))
                    cv2.imwrite(out_file, image)
            except:
                traceback.print_exc()
                print(image_file, flush=True)

    def search_face_task(self, bgr, thickness=2, fontScale=1.5, delay=0, vis=False):
        """
        1:N人脸搜索任务
        :param bgr: BGR image
        :return:
        """
        face_info = self.detect_search(bgr, max_face=-1, vis=False)
        image = self.draw_result("Recognizer", image=bgr, face_info=face_info,
                                 thickness=thickness, fontScale=fontScale, delay=delay, vis=vis)
        return image, face_info

    # 自定义
    def detect_image(self, image_path, vis=False):
        # image_list = file_utils.get_files_lists(image_dir, postfix=file_utils.IMG_POSTFIX)
        try:
            # print("image_file:{}\t".format(image_file), end=',', flush=True)
            image = image_utils.read_image_ch(image_path)
            image = image_utils.resize_image(image, size=(None, 640))
            image, face_info = self.search_face_task(image, vis=vis)
            if len(face_info['label']) > 0:
                if face_info['label'][0] == "unknown":
                    return 1, 0
                else:
                    return int(face_info['label'][0]), face_info['score'][0]
            else:
                return 1, 0
        except:
            traceback.print_exc()
            print(image_path, flush=True)
            return 1, 0


def parse_opt():
    database = configs.database  # 存储人脸数据库特征路径database
    # image_dir = 'data/database-test'  # 测试图片的目录
    image_dir = './assets/face/test'  # 测试图片的目录
    out_dir = "./assets/face/output/"  # 保存检测结果
    video_file = None  # video_file is None表示进行图片测试
    # video_file = "data/test-video.mp4"  # 视频文件测试
    # video_file = "0"  # 摄像头测试
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default=database, help='存储人脸数据库特征路径database')
    parser.add_argument('--image_dir', type=str, default=image_dir, help='image_dir')
    parser.add_argument('--video_file', type=str, default=video_file, help='camera id or video file')
    parser.add_argument('--out_dir', type=str, default=out_dir, help='save result')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == "__main__":
    """1:N人脸搜索,可用于人脸签到、人脸门禁、人员信息查询、安防监控等应用场景"""
    opt = parse_opt()
    fr = FaceModel(database=opt.database)
    # fr.create_database(portrait=configs.portrait, vis=False)
    fr.detect_image(r"./assets/face/test/test1_4.jpg", vis=True)

    # if isinstance(opt.video_file, str) or isinstance(opt.video_file, int):
    #     opt.video_file = str(opt.video_file)
    #     if len(opt.video_file) == 1: opt.video_file = int(opt.video_file)
    #     save_video = os.path.join(opt.out_dir, "result.avi") if opt.out_dir else None
    #     fr.start_capture(opt.video_file, save_video, detect_freq=1, vis=True)
    # else:
    #     fr.detect_image_dir(opt.image_dir, opt.out_dir, vis=True)
