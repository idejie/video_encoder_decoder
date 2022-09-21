import cv2
import subprocess
import time
import datetime
import numpy as np
'''拉流url地址，指定 从哪拉流'''
# video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 自己摄像头
pull_url = 'rtsp://localhost:8554/my' # "rtsp_address"
video_capture = cv2.VideoCapture(pull_url) # 调用摄像头的rtsp协议流
# pull_url = "rtmp_address"


'''推流url地址，指定 用opencv把各种处理后的流(视频帧) 推到 哪里'''
push_url = "rtsp://localhost:8554/lr"

width = 480
height = 270
fps = 60 # Error setting option framerate to value 0. 
print("width", width, "height", height,  "fps:", fps) 


# command = [r'D:\Softwares\ffmpeg-5.1-full_build\bin\ffmpeg.exe', # windows要指定ffmpeg地址
command = ['ffmpeg', # linux不用指定
    '-y', '-an',
    '-f', 'rawvideo',
    # '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24', #像素格式
    '-s', "{}x{}".format(width, height),
    '-r', str(fps), 
    '-i', '-',
    '-crf','18',
    '-c:v', 'libx264',  # 视频编码方式
    # '-pix_fmt', 'yuv420p',
    '-preset', 'veryfast',
    '-tune','zerolatency',
    # '-b:v', '4m',
    # '-profile:v','high',
    '-qp', '25',
    '-f', 'rtsp', #  flv rtsp
    '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
    push_url] # rtsp rtmp  
pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

def frame_handler(frame):
    ...
    return frame


process_this_frame = True 
frame = cv2.imread('010.jpg')

while True: # True or video_capture.isOpened():
    # Grab a single frame of video
    # ret, frame = video_capture.read()
    image = frame.copy()
    start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # print()
    # frame = cv2.resize(frame, (width,height), interpolation = cv2.INTER_AREA)
    # frame = np.zeors((width,height,3))
    # frame = np.int8(frame)
    # cv2.putText(image, str(start), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    # cv2.imwrite("frame_in.jpg", frame)

    # handle the video capture frame
    
    
    # frame = frame_handler(frame) 
    
    # Display the resulting image. linux 需要注释该行代码
    # cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(delay=100) & 0xFF == ord('q'): #  delay=100ms为0.1s .若dealy时间太长，比如1000ms，则无法成功推流！
    #     break
    
    # pipe.stdin.write(frame.tostring())
    pipe.stdin.write(image.tobytes())
    
# video_capture.release()
# cv2.destroyAllWindows()
pipe.terminate()
