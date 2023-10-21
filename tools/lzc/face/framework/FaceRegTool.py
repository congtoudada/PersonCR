
class FaceRegTool:
    @staticmethod
    def pack_req(obj_id: int, img) -> dict:
        """
        编码。用户调用

        参数:
        obj_id (int): 目标追踪id
        img (str/numpy array): 目标图像信息

        返回:
        dict
        """
        face_infer = {
            'obj_id': obj_id,
            'img': img
        }
        return face_infer

    @staticmethod
    def unpack_req(face_infer: dict):
        """
        解码。系统调用

        参数:
        face_result (dict): 结果信息dict

        返回:
        obj_id: 目标追踪id
        img: 目标图像信息
        """
        return face_infer['obj_id'], face_infer['img']

    @staticmethod
    def pack_rsp(obj_id: int, person_id: int, score: float, img) -> dict:
        """
        编码。系统调用

        参数:
        obj_id (int): 目标追踪id
        person_id (int): 识别结果，对应数据库主键。1表示陌生人
        score (float): 识别置信度
        img (ndarry): 人脸图片

        返回:
        dict
        """
        face_result = {
            'obj_id': obj_id,
            'person_id': person_id,
            'score': score,
            'face_img': img
        }
        return face_result

    @staticmethod
    def unpack_rsp(face_result: dict):
        """
        解码。用户调用

        参数:
        face_result (dict): 结果信息dict

        返回:
        obj_id: 目标追踪id
        person_id: 识别结果，对应数据库主键。-1表示未识别
        score: 预测分数
        img: 人脸图片
        """
        return face_result['obj_id'], face_result['person_id'], face_result['score'], face_result['face_img']
