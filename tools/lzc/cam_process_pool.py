from .cam_process import *

# ----------------------------- 客流1 -----------------------------
def write_keliu1(qframe, cam_event, esc_event, cam_yaml):
    write_process(qframe, cam_event, esc_event, cam_yaml)

def read_keliu1(qframe, cam_event, face_queue_list, sql_queue_list, esc_event, args, main_yaml, cam_yaml):
    read_process(qframe, cam_event, face_queue_list, sql_queue_list, esc_event, args, main_yaml, cam_yaml)

# ----------------------------- 人脸1 -----------------------------
def write_renlian1(qframe, cam_event, esc_event, cam_yaml):
    write_process(qframe, cam_event, esc_event, cam_yaml)

def read_renlian1(qframe, cam_event, face_queue_list, sql_queue_list, esc_event, args, main_yaml, cam_yaml):
    read_process(qframe, cam_event, face_queue_list, sql_queue_list, esc_event, args, main_yaml, cam_yaml)


