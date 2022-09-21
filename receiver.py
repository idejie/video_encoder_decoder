import cv2
import numpy as np
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
vcap = cv2.VideoCapture("rtsp://localhost:8554/my", cv2.CAP_FFMPEG)
while(1):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break;
    else:
        #cv2.imshow('VIDEO', frame)
        #cv2.waitKey(1)
        print(frame.shape)
        cv2.imwrite("frame_out_4m.jpg", frame)
