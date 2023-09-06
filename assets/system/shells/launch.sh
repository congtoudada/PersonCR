#!/bin/bash

# TensorRT安装路径
export LD_LIBRARY_PATH=/home/ps/TensorRT-8.5.3.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/ps/TensorRT-8.5.3.1/lib:$LIBRARY_PATH

# 进入工程目录
cd /home/ps/lzc/Bytetrack2
# 激活虚拟环境
source /home/ps/anaconda3/bin/activate bytetrack38

# 多卡运行
CUDA_VISIBLE_DEVICES=0 python tools/demo_track_process.py --trt --main main_process1 &
CUDA_VISIBLE_DEVICES=1 python tools/demo_track_process.py --trt --main main_process2
