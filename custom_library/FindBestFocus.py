import cv2
import numpy as np

thres = 0.5

for i in range(11):
    img = cv2.imread('focustest/f%02d.jpg'%(i))
    focus_area = img[2300:2700,1600:2000]

    focus_area = cv2.cvtColor(focus_area,cv2.COLOR_BGR2GRAY)
    df = focus_area.copy()
    df = df - np.min(df)
    df = df * (1.0/np.max(df))
    print(np.min(df),np.max(df), df.dtype)
    dfy = cv2.Sobel(df,-1,0,1,ksize=3)
    dfx = cv2.Sobel(df,-1,1,0,ksize=3)
    df = dfx**2 + dfy**2
    print(np.min(df),
        np.max(df),
        np.mean(df),
        np.mean(df>thres*np.std(df) + np.mean(df)))

    cv2.imshow('full',cv2.resize(img,(0,0),fx=0.1,fy=0.1))
    cv2.imshow('ROIorigin',img[2300:2700,1600:2000])
    cv2.imshow('ROI',focus_area*(df>thres*np.std(df) + np.mean(df))+(focus_area*0.6).astype(np.uint8))
    cv2.waitKey()