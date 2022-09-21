#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import queue
import threading
import torch
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import matplotlib.pyplot as plt
from PIL import Image
import subprocess


import cv2
import time
from multiprocessing import Process
TRT_LOGGER = trt.Logger()

hr_url = "rtsp://localhost:8554/hd2"
lr_url = "rtsp://localhost:8554/lr2"

engine_file = "./encoder.trt"
width = 480
height = 270
fps = 25 # Error setting option framerate to value 0. 
print("width", width, "height", height,  "fps:", fps) 

# command = [r'D:\Softwares\ffmpeg-5.1-full_build\bin\ffmpeg.exe', # windows要指定ffmpeg地址
command = ['/usr/bin/ffmpeg', # linux不用指定
    '-y', '-an',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24', #像素格式
    '-s', "{}x{}".format(width, height),
    '-r', str(fps), 
    '-i', '-',
    '-crf','18',
    '-c:v', 'libx264',  # 视频编码方式
    '-pix_fmt', 'yuvj444p',
    '-preset', 'veryfast',
    '-tune','zerolatency',
    # '-b:v', '4m',
    '-profile:v','high444',
    # '-qp', '1',
    '-f', 'rtsp', #  flv rtsp
    '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
    lr_url] # rtsp rtmp  
pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

q = queue.Queue()
# 帧号
frame_num = 0
vcap = cv2.VideoCapture(hr_url, cv2.CAP_FFMPEG)
def get_frame():
    # 生产者不断获取图像
    global frame_num
    
    while True:
        ret, image = vcap.read()
        frame_num +=1
        if ret:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('encoder_input.jpg', image)
            data = (np.asarray(img).astype('float32') / float(255.0))
            input_image = np.moveaxis(data, 2, 0)
            data = {'input_image':input_image,'frame_num':frame_num}
            # print(f'receive {frame_num}',end='\r')
        else:
            print('[Error] Empoty Frame')
            data = {'input_image':None,'frame_num':frame_num}
        q.put(data)
        # time.sleep(1)


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def consumer():
    # import pycuda.driver as cuda
    # import pycuda.autoinit
    # 消费者不断压缩图像并
    engine =  load_engine(engine_file)
    # cuda.init()
    # device = cuda.Device(0)  # enter your Gpu id here
    # ctx = device.make_context()
    context = engine.create_execution_context()
    image_height,image_width = 1080,1920
    input_image = np.zeros((image_height,image_width,3))
    data = (np.asarray(input_image).astype('float32') / float(255.0))
                
    input_image = np.moveaxis(data, 2, 0)
    context.set_binding_shape(engine.get_binding_index("input_1"), (1, 3, image_height, image_width))
    # Allocate host and device buffers
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            input_buffer = np.ascontiguousarray(input_image)
            input_memory = cuda.mem_alloc(input_image.nbytes)
            bindings.append(int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))

    stream = cuda.Stream()
        # print("Reading input image from file {}".format(input_file))
        # img = input_file
    while(1):
        # global context, stream, output_buffer
        # global input_memory
        data_in = q.get()
        if  data_in['input_image'] is None:
            print("[Error]: Frame is empty")
        else:
            # if data_in['frame_num']>10000:
            #     break
            # engine =  load_engine(engine_file)
            # context = engine.create_execution_context()
            # image_height,image_width = 1080,1920
            # input_image = np.zeros((image_height,image_width,3))
            # data = (np.asarray(input_image).astype('float32') / float(255.0))
                        
            # input_image = np.moveaxis(data, 2, 0)
            # context.set_binding_shape(engine.get_binding_index("input_1"), (1, 3, image_height, image_width))
            # # Allocate host and device buffers
            # bindings = []
            # for binding in engine:
            #     binding_idx = engine.get_binding_index(binding)
            #     size = trt.volume(context.get_binding_shape(binding_idx))
            #     dtype = trt.nptype(engine.get_binding_dtype(binding))
            #     if engine.binding_is_input(binding):
            #         input_buffer = np.ascontiguousarray(input_image)
            #         input_memory = cuda.mem_alloc(input_image.nbytes)
            #         bindings.append(int(input_memory))
            #     else:
            #         output_buffer = cuda.pagelocked_empty(size, dtype)
            #         output_memory = cuda.mem_alloc(output_buffer.nbytes)
            #         bindings.append(int(output_memory))
            # import pycuda.autoinit
            # stream = cuda.Stream()
            # start = time.time()
            #cv2.imshow('VIDEO', frame)
            #cv2.waitKey(1)
            # print(image.shape)
            image =  data_in['input_image']
            # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(img.shape)
            # cv2.imwrite('encoder_input.jpg', image)
            # data = (np.asarray(img).astype('float32') / float(255.0))
            
            # input_image = np.moveaxis(data, 2, 0)
            input_buffer = np.ascontiguousarray(image)
            # end = time.time()
            # print('input_buffer',(end-start)*1000)
                # Set input shape based on image dimensions for inference
        
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # end = time.time()
            # print('memcpy_htod_async',(end-start)*1000)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            # print('prediction',(end-start)*1000)
            # Synchronize the stream
            stream.synchronize()
            # end = time.time()
            # print('synchronize',(end-start)*1000)
            # print(output_buffer.shape)
            data = np.reshape(output_buffer, (3, image_height // 4, image_width // 4))* 255.
            # print(data,)
            # data = np.moveaxis(data, 0, 2)
            # data = data 
            # data = data.astype(int)
            # data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            # data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            # print(data.shape,data_in['frame_num'])
            
            # cv2.imwrite('encoder.jpg', data.astype(np.uint8))
            # # data = cv2.imread('encoder.jpg')
            # end = time.time()
            # print('all',(end-start)*1000)
            pipe.stdin.write(data.astype(np.uint8).tobytes())
          
# for j in range(2):
t = threading.Thread(target=get_frame)
t.start()
# t = threading.Thread(target=get_frame)
# t.start()
time.sleep(10)
# # # 2
# for j in range(1):
#     v = threading.Thread(target=consumer)
#     v.start()
# c1 = Process(target=consumer)
# c1.start()
consumer()
# pipe.terminate()