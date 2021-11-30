import cv2
import numpy as np

img = cv2.imread('focustest.jpg')
ROI_pivot = (280,280) # H, W
ROI_size = (300,300)

ROI_img = img[ROI_pivot[0]:ROI_pivot[0]+ROI_size[0],
        ROI_pivot[1]:ROI_pivot[1]+ROI_size[1]]
ROI_img = cv2.cvtColor(ROI_img,cv2.COLOR_BGR2GRAY)
scharr_kernel = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
ROI_img_dx = cv2.filter2D(ROI_img.astype(np.float32), -1,scharr_kernel)
print(ROI_img_dx.dtype)
print(np.min(ROI_img_dx), np.max(ROI_img_dx))

# normalize
ROI_img_dx -= (np.max(ROI_img_dx)+np.min(ROI_img_dx))/2
print(np.min(ROI_img_dx), np.max(ROI_img_dx))
ROI_img_bdx = np.abs(ROI_img_dx)
ROI_img_bdx /= np.max(ROI_img_bdx)

# threshold
ROI_img_bdx *= (ROI_img_bdx > 0.5)

# to uint8 img
ROI_img_bdx *= 255
ROI_img_bdx = ROI_img_bdx.astype(np.uint8)
cv2.imshow('title1', img[ROI_pivot[0]:ROI_pivot[0]+ROI_size[0],
        ROI_pivot[1]:ROI_pivot[1]+ROI_size[1]])
cv2.imshow('title0',ROI_img_bdx)
cv2.waitKey()