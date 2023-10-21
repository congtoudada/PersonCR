
class CountMgrData:
    def __init__(self, init_dict):
        self.init_dict = init_dict
        # cam_yaml
        self.run_mode = init_dict['run_mode']  # 0：客流 1：人脸
        self.cam_name = init_dict['cam_name']  # 相机名称
        self.cal_mode = init_dict['cal_mode']  # 算法模式 0:左上角为基准点 1:中心为基准点
        self.det_mode = init_dict['det_mode']  # 检测模式 0:常规 1:里外包围盒
        self.cam_id = init_dict['cam_id']  # 相机id
        self.center_ratio = init_dict['center_ratio']  # 中线位置比例 (0:up_ratio 1:down_ration)
        self.up_ratio = init_dict['up_ratio']  # 上部：检测区域
        self.down_ratio = init_dict['down_ratio']  # 下部：检测区域
        self.buffer_up_ratio = init_dict['buffer_up_ratio']  # 上部：缓冲区域
        self.buffer_down_ratio = init_dict['buffer_down_ratio']  # 下部：缓冲区域
        self.left_ratio = init_dict['left_ratio']  # 左部：检测区域
        self.right_ratio = init_dict['right_ratio']  # 右部：检测区域
        self.buffer_left_ratio = init_dict['buffer_left_ratio']  # 左部：缓冲区域
        self.buffer_right_ratio = init_dict['buffer_right_ratio']  # 右部：缓冲区域
        self.is_vertical = init_dict['is_vertical']  # 是否竖直检测
        self.is_reverse = init_dict['is_reverse']  # 是否反向 [以竖直方向为例：默认:往上离开(Out) 反向:往下离开(Out)]
        self.only_dir = init_dict['only_dir']  # 是否只检测一个方向 0:双向检测 1:只检测进 2:只检测出
        self.is_vis = init_dict['is_vis']  # 是否可视化
        # 抓拍设置
        self.is_img = init_dict['is_img']  # 是否抓拍图像
        self.border = init_dict['border']  # 抓拍边框大小，该值越大抓拍尺寸越大
        self.is_vid = init_dict['is_vid']  # 是否抓拍视频
        self.cap_width = init_dict['cap_width']  # 抓拍宽
        self.cap_height = init_dict['cap_height']  # 抓拍高
        self.cap_fps = init_dict['cap_fps']  # 抓拍帧率
        self.save_frames = init_dict['save_frames']  # 抓拍视频时，保留的额外帧数
        self.max_record_frames = init_dict['max_record_frames']  # 抓拍视频上限帧数
        self.lost_frames = init_dict['lost_frames']  # 对象消失多少帧销毁

        # main_yaml
        self.debug_mode = init_dict['debug_mode']  # debug模式
        self.frame_history_capacity = init_dict['frame_history_capacity']  # 回放视频帧缓存上限
        self.reg_interval = init_dict['reg_interval']  # 人脸识别频率
        self.reg_count_thresh = init_dict['reg_count_thresh']  # 人脸识别决定阈值

        # 其他
        self.save_path = init_dict['save_path']
        self.qface_req = init_dict['qface_req']  # 进程队列：人脸请求
        self.qface_rsp = init_dict['qface_rsp']  # 进程队列：人脸响应
        self.qsql_list = init_dict['qsql_list']  # 进程队列：数据库



class CountMgrRunningData:
    def __init__(self):
        self.obj_id = 0
        self.point = (0, 0)
        self.score = 0
        self.im = None
        self.history_im = None
        self.tlwh = (0, 0, 0, 0)
        self.now = 0
        self.frame_id = 0

    def init(self, obj_id, point, score, im, history_im, tlwh, now, frame_id):
        self.obj_id = obj_id
        self.point = point
        self.score = score
        self.im = im
        self.history_im = history_im
        self.tlwh = tlwh
        self.now = now
        self.frame_id = frame_id
