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
TRT_LOGGER = trt.Logger()

hr_url = "rtsp://localhost:8554/hd2"
lr_url = "rtsp://localhost:8554/lr"

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


def preprocess(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = (np.asarray(img).astype('float32') / float(255.0))
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def postprocess(data, path):
    data = np.moveaxis(data, 0, 2)
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, data * 255.)
    return data

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    with load_engine(engine_file) as engine:
        # print("Reading input image from file {}".format(input_file))
        # img = input_file
       
        vcap = cv2.VideoCapture(hr_url, cv2.CAP_FFMPEG)
        while(1):
            ret, image = vcap.read()
            if ret == False:
                print("[Error]: Frame is empty")
                break
            else:
                #cv2.imshow('VIDEO', frame)
                #cv2.waitKey(1)
                # print(image.shape)
                
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imwrite('encoder_input.jpg', image)
                data = (np.asarray(img).astype('float32') / float(255.0))
                
                input_image = np.moveaxis(data, 2, 0)
                image_width = input_image.shape[-1]
                image_height = input_image.shape[-2]

                with engine.create_execution_context() as context:
                    # Set input shape based on image dimensions for inference
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
                    # Transfer input data to the GPU.
                    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
                    # Run inference
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    # Transfer prediction output from the GPU.
                    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                    # Synchronize the stream
                    stream.synchronize()
                # print(output_buffer.shape)
                data = np.reshape(output_buffer, (3, image_height // 4, image_width // 4))
                # print(data,)
                data = np.moveaxis(data, 0, 2)
                data = data * 255.
                # data = data.astype(int)
                # data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite('encoder.jpg', data.astype(np.uint8))
                # data = cv2.imread('encoder.jpg')
                
                pipe.stdin.write(data.astype(np.uint8).tobytes())

                

                # postprocess(np.reshape(output_buffer, (3, image_height // 4, image_width // 4)), output_file)
                # print("Writing output image to file {}".format(output_file))
        pipe.terminate()

# def main():
#     input_file = "010.jpg"
#     output_file = "out.png"

#     # img = cv2.imread('./test.png')
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # print(img.shape)

#     print(f"Running TensorRT inference for {engine_file}")
#     with load_engine(engine_file) as engine:
#         infer(engine, input_file, output_file)

if __name__ == '__main__':
    main()