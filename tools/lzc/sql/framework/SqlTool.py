import os
from enum import Enum


class SqlTool:

    @staticmethod
    def pack_test_data(password, description):
        # table_name = "user"
        sqlDict = {
            "password": password,
            "description": description,
        }
        return {'run_mode': -1, 'data': sqlDict}

    @staticmethod
    def pack_keliu_data(record_time, flow_cam_id, record_status,
                        record_num, record_photo_url, is_warning, record_video_url):
        # table_name = "flow_record"
        if record_photo_url is not None and record_photo_url != "":
            record_photo_url = os.path.join(*record_photo_url.split('/')[1:])
        record_video_url = os.path.join(*record_video_url.split('/')[1:])
        sqlDict = {
            "record_time": record_time,
            "flow_cam_id": flow_cam_id,
            "record_status": record_status,
            "record_num": record_num,
            "record_photo_url": record_photo_url,
            "is_warning": is_warning,
            "record_video_url": record_video_url,
        }
        return {'run_mode': 0, 'data': sqlDict}

    @staticmethod
    def pack_renlian_data(record_time, recognize_cam_id, record_status, record_num,
                          record_photo_url, personnel_id, is_warning, record_video_url):
        # table_name = "recognize_record"
        if record_photo_url is not None and record_photo_url != "":
            record_photo_url = os.path.join(*record_photo_url.split('/')[1:])
        record_video_url = os.path.join(*record_video_url.split('/')[1:])
        sqlDict = {
            "record_time": record_time,
            "recognize_cam_id": recognize_cam_id,
            "record_status": record_status,
            "record_num": record_num,
            "record_photo_url": record_photo_url,
            "personnel_id": personnel_id,
            "is_warning": is_warning,
            "record_video_url": record_video_url,
        }
        return {'run_mode': 1, 'data': sqlDict}

    @staticmethod
    def unpack_data(data: dict) -> (int, dict):
        return data['run_mode'], data['data']

