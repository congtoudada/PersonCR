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
from Face_Recognition.core import face_recognizer
from pybaseutils import image_utils, file_utils


class Example2(face_recognizer.FaceRecognizer):
    def __init__(self, database):
        """
        @param database: 人脸数据库的路径
        """
        super(Example2, self).__init__(database=database)

    def compare_face_task(self, image_file1, image_file2, score_thresh=0.75, vis=True):
        """
        1:1人脸比对,比较两张肖像图是否是同一个人
        @param image_file1 肖像图1
        @param image_file2 肖像图2
        @param score_thresh 相似人脸分数人脸阈值
        """
        # 读取图片
        # image1 = cv2.imread(image_file1)
        # image2 = cv2.imread(image_file2)
        image1 = image_utils.read_image_ch(image_file1)
        image2 = image_utils.read_image_ch(image_file2)
        face_info1, face_info2, score = self.compare_face(image1, image2)
        if len(face_info1['face']) > 0 and len(face_info2['face']) > 0:
            v1 = face_info1["feature"]
            v2 = face_info2["feature"]
            same_person = score > score_thresh
            print("feature1.shape:{}\nfeature1:{}".format(v1.shape, v1[0, 0:20]))
            print("feature2.shape:{}\nfeature2:{}".format(v2.shape, v2[0, 0:20]))
            print("similarity: {}, same person: {}".format(score, same_person))
            if vis: self.show_result(image1, face_info1, image2, face_info2)
        else:
            print("No face detected")
        return score

    def show_result(self, image1, face_info1, image2, face_info2):
        face1 = face_info1["face"]
        face2 = face_info2["face"]
        if len(face1) > 0: image_utils.cv_show_image("face1", face1[0], delay=1)
        if len(face2) > 0: image_utils.cv_show_image("face2", face2[0], delay=1)
        self.draw_result("image1", image=image1, face_info=face_info1, vis=True, delay=1)
        self.draw_result("image2", image=image2, face_info=face_info2, vis=True, delay=0)


def parse_opt():
    image_file1 = "data/test_image/test1.jpg"
    image_file2 = "data/test_image/test2.jpg"
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file1', type=str, default=image_file1, help='image_file1')
    parser.add_argument('--image_file2', type=str, default=image_file2, help='image_file1')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == "__main__":
    """1:1人脸比对,可用于人证比对等场景"""
    opt = parse_opt()
    fr = Example2(database="")
    fr.compare_face_task(opt.image_file1, opt.image_file2, vis=True)
