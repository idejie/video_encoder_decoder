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
#import matplotlib.pyplot as plt
#from PIL import Image
import subprocess
import cv2
import time
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor
# TRT_LOGGER = trt.Logger()

hr_url = "rtsp://192.168.31.228:8554/hd2"
lr_url = "rtsp://61.174.252.126:8554/lr"
image_height,image_width = 270,480
engine_file = "/home/nvidia/Downloads/decoder_fp16.trt"
width = 480
height = 270
fps = 25 # Error setting option framerate to value 0. 
print("width", width, "height", height,  "fps:", fps) 

# command = [r'D:\Softwares\ffmpeg-5.1-full_build\bin\ffmpeg.exe', # windows要指定ffmpeg地址
command = ['ffmpeg', # linux不用指定
    '-y', '-an',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24', #像素格式
    '-s', "{}x{}".format(width, height),
    '-r', str(fps), 
    '-i', '-',
    #'-crf','18',
    '-c:v', 'libx264',  # 视频编码方式
    '-pix_fmt', 'yuvj444p',
    '-preset', 'veryfast',
    '-tune','zerolatency',
    #'-b:v', '4m',
    '-profile:v','high444',
    '-qp', '20',
    '-f', 'rtsp', #  flv rtsp
    '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
    hr_url] # rtsp rtmp  
pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

q = queue.Queue()
# 帧号
frame_num = 0
vcap = cv2.VideoCapture(lr_url, cv2.CAP_FFMPEG)
ret, image = True, cv2.imread('encoder.jpg')
def get_frame():
    # 生产者不断获取图像
    global frame_num
    while True:
        frame_num +=1
        if ret:
            # print('[Info] right Frame')
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('encoder_input.jpg', image)
            #print(image.shape)
            data = (np.asarray(img).astype('float32') / float(255.0))
            input_image = np.moveaxis(data, 2, 0)
            data = {'input_image':input_image,'frame_num':frame_num}
            # print(f'receive {frame_num}',end='\r')
        else:
            print('[Error] Empoty Frame')
            data = {'input_image':None,'frame_num':frame_num}
        q.put(data)
        # print(data['input_image'].shape)
        # time.sleep(1)


start  = time.time()
class TRTInference:
    def __init__(self, trt_engine_datatype, batch_size, trt_engine_path="/home/nvidia/Downloads/decoder_fp16.trt"):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs  = []
        cuda_inputs  = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        self.stream  = stream
        self.context = context
        self.engine  = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings


    def infer(self, image):
        
        threading.Thread.__init__(self)
        start_time = time.time()
        self.cfx.push()
        # restore
        stream  = self.stream
        context = self.context
        engine  = self.engine

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # read image
        # image = 1 - (np.asarray(Image.open(input_img_path), dtype=np.float)/255)
        np.copyto(host_inputs[0],  image.ravel())

        # inference
        #start_time = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        #print('execute time', 1000*(time.time()-start_time))
  
        # parse output
        start_time = time.time()
        data = np.reshape(host_outputs[0], (3, image_height* 4, image_width*4))* 255.
        data = np.zeros((3,480,270))
        start_time = time.time()
        pipe.stdin.write(data.astype(np.uint8).tobytes())
        #print('execute time', 1000*(time.time()-start_time))
        # output = np.array([math.exp(o) for o in host_outputs[0]])
        # output /= sum(output)
        # for i in range(len(output)): print("%d: %.2f"%(i,output[i]))

        self.cfx.pop()


    def destory(self):
        self.cfx.pop()


exitFlag = 0

class myThread(threading.Thread):
   def __init__(self, func, args):
      threading.Thread.__init__(self)
      self.func = func
      self.args = args
   def run(self):
      print ("Starting " , self.args[0])
    #   self.func(self.args[1])
      print ("Exiting " , self.args[0])



max_batch_size = 1
trt_inference_wrapper = TRTInference(trt_engine_path=engine_file,
        trt_engine_datatype=trt.DataType.FLOAT,
        batch_size=max_batch_size)


# t = threading.Thread(target=get_frame)
# t.start()
# for j in range(2):
t = threading.Thread(target=get_frame)
t.start()
#t = threading.Thread(target=get_frame)
#t.start()
# t.join()
time.sleep(3)
# def run(args):
#     print ("Starting " , args[0])
#   self.func(self.args[1])
#     print ("Exiting " , args[0])
executor = ThreadPoolExecutor(max_workers=10)
print('eee')
print(q.qsize())

while True:
    #if  q.empty():
         # print('eee')
        #print('eeee',q.qsize())
    #    continue
    # print('eee')
    # print('q_size',q.qsize())
    data_in = q.get()
    
    if  data_in['input_image'] is None:
        #print("[Error]: Frame is empty")
        continue
    else:
        # thread1 = myThread(trt_inference_wrapper.infer, [data_in['frame_num'],data_in['input_image']])
        # 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞
        task1 = executor.submit(trt_inference_wrapper.infer, (data_in['input_image']))
        # task2 = executor.submit(get_html, (2))



# pipe.terminate()
