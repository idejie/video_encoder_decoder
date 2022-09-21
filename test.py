import cv2

import numpy as np

in1= cv2.imread('encoder.jpg')

in2 =cv2.imread('decoder_input.jpg')
print(in2.astype(np.float32)-in1.astype(np.float32))
cv2.imwrite('error.jpg',(in2-in1))
print(np.sum(np.abs(in2-in1))/(3*480*270*16))