import os
from enum import Enum

class SqlHelper:

    def packTestData(self, password, description):
        # table_name = "user"
        sqlDict = {
            "password": password,
            "description": description,
        }
        return {'run_mode': -1, 'sqlDict': sqlDict}

    def packKeliuData(self, record_time, flow_cam_id, record_status,
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
        return {'run_mode': 0, 'sqlDict': sqlDict}

    def packRenlianData(self, record_time, recognize_cam_id, record_status, record_num,
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
        return {'run_mode': 1, 'sqlDict': sqlDict}

    def unpackData(self, data: dict):
        return data['run_mode'], data['sqlDict']

