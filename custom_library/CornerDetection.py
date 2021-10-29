import numpy as np
import cv2


def findHorizontalLine(img):
    # finds horizontal edge line in the top right corner cropped image.
    # image should be of shape (512,512), any type or range possible
    # to maximize performance it is devided into 2 steps:
    #   rough search : uses reduced image(of size 2**6) and 32 steps
    #   finds line position within +-2**3 pixel error in original image
    # 
    #   detailed search : uses original image and 8 steps
    #   findes line position within +-2 pixel error
    # 
    # returns (bias, right_bias) where
    #   bias is line's intercept with left edge of the image
    #   right_bias is the intercept with right edge of the image


    img = img.astype(np.float32)/255
    ################
    img_rsz = cv2.resize(img,dsize=(2**6,2**6))

    dy_img = cv2.Sobel(img,-1,0,1,ksize=5)
    dy_img_rsz = cv2.Sobel(img_rsz,-1,0,1)

    bias = np.argmax(np.sum(dy_img[:,:5],axis=1))
    bias_rsz = np.argmax(dy_img_rsz[:,0])

    x = np.arange(0,2**6,1).reshape((2**6,1))
    x_shifted = np.ones((2**5+1,2**6,2**6))*x - bias_rsz

    end_points = np.arange(0,2**5+1,1)*2 - bias_rsz
    b_shift = np.arange(0,2**6,1)

    for i in range(2**5+1):
        x_shifted[i] -= b_shift*end_points[i]/(2**6)

    g = np.e**(-(x_shifted**2)*0.5/(2**2))

    linear_filtered = g*dy_img_rsz

    right_bias = np.argmax(np.sum(linear_filtered,axis=(1,2)))*2**4

    ######
    clip_range = np.sort([2**4,bias,right_bias,2**9-2**4])
    clip_from = clip_range[1] - 2**4
    clip_to = clip_range[2] + 2**4

    end_points = np.arange(-2**2-1,2**2+1,1)-bias+right_bias
    b_shift = np.arange(0,2**9,1)

    x = np.arange(0,clip_to - clip_from,1).reshape((clip_to - clip_from,1))
    x_shifted = np.ones((2**3+1,clip_to - clip_from,2**9))*x - bias + clip_from

    for i in range(2**3+1):
        x_shifted[i] -= b_shift*end_points[i]/(2**9)

    g = np.e**(-(x_shifted**2)*0.5/2)

    linear_filtered2 = g*dy_img[clip_from:clip_to,:]

    right_bias += np.argmax(np.sum(linear_filtered2,axis=(1,2)))-(2**2+1)

    #######

    return (bias, right_bias)