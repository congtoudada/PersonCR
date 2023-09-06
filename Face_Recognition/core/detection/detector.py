# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-12-31 09:11:25
# --------------------------------------------------------
"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import cv2
from pybaseutils import image_utils, file_utils


class FaceDetector(object):
    def __init__(self, net_name, input_size, conf_thresh=0.5, nms_thresh=0.3, device="cuda:0"):
        top_k = 500
        keep_top_k = 750
        if net_name.lower() == "RFB".lower():
            from Face_Recognition.core.detection.light_detector.light_detector import UltraLightFaceDetector
            self.detector = UltraLightFaceDetector(model_file="",
                                                   net_name=net_name,
                                                   input_size=input_size,
                                                   conf_thresh=conf_thresh,
                                                   iou_thresh=nms_thresh,
                                                   top_k=top_k,
                                                   keep_top_k=keep_top_k,
                                                   device=device)
        elif net_name.lower() == "mtcnn".lower():
            from Face_Recognition.core.detection.mtcnn.mtcnn_detector import MTCNNDetector
            self.detector = MTCNNDetector(device=device)
        else:
            raise Exception("unsupported net_name:{}".format(net_name))
        print("net_name  :{}".format(net_name))
        print("use device:{}".format(device))

    def detect_face_landmarks(self, bgr, vis=False):
        """
        :param bgr:
        :return:
        """
        bboxes, scores, landms = self.detector.detect(bgr, vis)
        return bboxes, scores, landms

    def detect_image_dir(self, image_dir, vis=True):
        image_list = file_utils.get_files_lists(image_dir)
        for image_file in image_list:
            image = cv2.imread(image_file)
            bboxes, scores, landms = self.detect_face_landmarks(image, vis=vis)
            print("bboxes:\n{}\nscores:\n{}\nlandms:\n{}".format(bboxes, scores, landms))

    @staticmethod
    def show_landmark_boxes(title, image, bboxes, scores, landms):
        """
        显示landmark和boxes
        :param title:
        :param image:
        :param landms: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        """
        image = image_utils.draw_landmark(image, landms, radius=2, vis_id=False)
        text = scores.reshape(-1).tolist()
        text = ["{:3.3f}".format(t) for t in text]
        image = image_utils.draw_image_bboxes_text(image, bboxes, text, thickness=2, fontScale=1.0, color=(255, 0, 0))
        image_utils.cv_show_image(title, image)
        return image


if __name__ == '__main__':
    image_dir = "test.jpg"
    image_dir = "../../data/test_image"
    input_size = [320, None]
    device = "cuda:0"
    detector = FaceDetector(net_name="mtcnn",
                            input_size=input_size,
                            device=device)
    detector.detect_image_dir(image_dir, vis=True)
