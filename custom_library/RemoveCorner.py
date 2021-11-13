import numpy as np
import cv2
import matplotlib.pyplot as plt


card = cv2.imread('creditcard.jpg')


card = cv2.cvtColor(card,cv2.COLOR_BGR2BGRA)
H, W = card.shape[:2]
d = int(H*0.06)
corner_filter = np.ones((H,W),dtype=np.uint8)

for i in range(d):
    corner_filter[i,:d-i] = 0
    corner_filter[i,W-d+i-1:] = 0
    corner_filter[H-i-1,:d-i] = 0
    corner_filter[H-i-1,W-d+i-1:] = 0

alpha = (np.mean(card[:,:,:3],axis=2)<230) | corner_filter

card[:,:,3] *= alpha


cv2.imshow('title', card)
cv2.waitKey()
cv2.imwrite('newcard.png',card)