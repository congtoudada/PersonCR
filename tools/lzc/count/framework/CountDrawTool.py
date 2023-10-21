import cv2

from tools.lzc.count.framework.CountMgrData import CountMgrData


class CountDrawTool:
    # 绘制计数结果
    @staticmethod
    def draw_count(countMgr, im):
        cv2.putText(im, "In: %d - Out: %d" % (countMgr.in_count, countMgr.out_count),
                    (8, im.shape[0] - int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                    thickness=2)

    @staticmethod
    def draw_zone(countMgr, im):
        im_h, im_w = im.shape[:2]
        text_scale = 2
        text_thickness = 2
        line_thickness = 2
        data: CountMgrData = countMgr.data
        if data.is_vertical:  # 是否竖直检测
            # 绘制检测区域
            zone_color = (118, 154, 242)  # 朱颜酡 cv2: BGR (118, 154, 242)
            cv2.rectangle(im, pt1=(int(data.left_ratio * im_w), int(data.up_ratio * im_h)),
                          pt2=(int(data.right_ratio * im_w), int(data.down_ratio * im_h)), color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Detection", (0, int(data.up_ratio * im_h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        zone_color,
                        thickness=text_thickness)
            # 绘制检测区域-中线
            if data.det_mode == 0:
                zone_color = (130, 107, 220)  # 长春 BGR
                startPoint = (int(data.left_ratio * im_w), int(countMgr.half_ratio * im_h))
                endPoint = (int(data.right_ratio * im_w), int(countMgr.half_ratio * im_h))
                cv2.line(im, startPoint, endPoint, zone_color, thickness=2)

            # 绘制缓冲区域
            zone_color = (182, 182, 178)  # 月魄 BGR (182, 182, 178)
            cv2.rectangle(im, pt1=(int(data.buffer_left_ratio * im_w), int(data.buffer_up_ratio * im_h)),
                          pt2=(int(data.buffer_right_ratio * im_w), int(data.buffer_down_ratio * im_h)),
                          color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Buffer", (0, int(data.buffer_up_ratio * im_h)), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, zone_color,
                        thickness=text_thickness)
        else:
            # 绘制检测区域
            zone_color = (118, 154, 242)  # 朱颜酡 cv2: BGR (118, 154, 242)
            cv2.rectangle(im, pt1=(int(data.up_ratio * im_w), int(data.left_ratio * im_h)),
                          pt2=(int(data.down_ratio * im_w), int(data.right_ratio * im_h)), color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Detection", (int(data.up_ratio * im_w), int(text_scale * 20)),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, zone_color,
                        thickness=text_thickness)

            # 绘制检测区域-中线
            if data.det_mode == 0:
                zone_color = (130, 107, 220)  # 长春 BGR
                startPoint = (int(countMgr.half_ratio * im_w), int(data.left_ratio * im_h))
                endPoint = (int(countMgr.half_ratio * im_w), int(data.right_ratio * im_h))
                cv2.line(im, startPoint, endPoint, zone_color, thickness=2)

            # 绘制缓冲区域
            zone_color = (182, 182, 178)  # 月魄 BGR (182, 182, 178)
            cv2.rectangle(im, pt1=(int(data.buffer_up_ratio * im_w), int(data.buffer_left_ratio * im_h)),
                          pt2=(int(data.buffer_down_ratio * im_w), int(data.buffer_right_ratio * im_h)),
                          color=zone_color,
                          thickness=line_thickness)
            cv2.putText(im, "Buffer", (int(data.buffer_up_ratio * im_w), int(text_scale * 20)),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, zone_color,
                        thickness=text_thickness)