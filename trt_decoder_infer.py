import torch
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import matplotlib.pyplot as plt
from PIL import Image
import cv2

TRT_LOGGER = trt.Logger()

engine_file = "./decoder.trt"


def preprocess(image):
    img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def infer(engine, input_file, output_file):
    print("Reading input image from file {}".format(input_file))
    # img = input_file
    input_image = preprocess(input_file)
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

    postprocess(np.reshape(output_buffer, (3, image_height * 4, image_width * 4)), output_file)
    print("Writing output image to file {}".format(output_file))


def main():
    input_file = "encoder.jpg"
    output_file = "sr2.png"

    # img = cv2.imread('./test.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)

    print(f"Running TensorRT inference for {engine_file}")
    with load_engine(engine_file) as engine:
        infer(engine, input_file, output_file)


if __name__ == '__main__':
    main()
