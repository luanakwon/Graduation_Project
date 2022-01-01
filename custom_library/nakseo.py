import cv2
import numpy as np
import mediapipe as mp
import time


img = cv2.imread('shba_hand.png')
t0 = time.time()
input_pts = np.float32([[200,100],[100,200],[200,300],[300,200]])
output_pts = np.float32([[0,0],[0,200],[200,200],[200,0]])
M = cv2.getPerspectiveTransform(input_pts,output_pts)
img2 = cv2.warpPerspective(img,M,(201,201),flags=cv2.INTER_LINEAR)
print(f'big input {time.time()-t0}')

t0 = time.time()
img_small = img[100:300,100:300]
input_pts -= 100
M = cv2.getPerspectiveTransform(input_pts,output_pts)
img3 = cv2.warpPerspective(img_small,M,(201,201),flags=cv2.INTER_LINEAR)
print(f'small input {time.time()-t0}')


cv2.imshow('original',img)
cv2.imshow('tilted',img2)
cv2.imshow('tilted2',img3)
cv2.waitKey()