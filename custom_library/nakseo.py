import numpy as np
import cv2
import CardDetection

img_size = (480,640) # W,H
p_c = 0.4 # size of gray grid line in length percent
p_w = 0.2 # size of edge detection window, ratio wrt grid center cell 

# use webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# cordinate required to draw gray grid line
crop_window, _ = CardDetection.get_cw(img_size,p_c,p_w)

# video save settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('card_detect_result.mp4',fourcc,30.0,img_size)

while cv2.waitKey(33) < 0:
    # read image from webcam
    ret, img = cap.read()
    # mirror image
    img = cv2.flip(img,flipCode=1)
    # change to portrait of ratio 800:600
    img = cv2.resize(img[:,140:500],dsize=img_size)

    # all calculation in 1 channel grayscale img
    grimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # draw gray grid
    img[crop_window[0][0],:] = (150,150,150)
    img[crop_window[1][0],:] = (150,150,150)
    img[:,crop_window[0][1]] = (150,150,150)
    img[:,crop_window[1][1]] = (150,150,150)

    # detect card that fits in center grid cell
    # ret = False if no card detected
    # corner in clockwise order, starting from top left
    ret, corners = CardDetection.run(grimg,p_c,p_w)
    
    # draw card contour
    if ret:
        print(f'{ret}, {corners}')
        img = cv2.line(img,corners[0],corners[1],(0,255,255),1)
        img = cv2.line(img,corners[1],corners[2],(0,255,255),1)
        img = cv2.line(img,corners[2],corners[3],(0,255,255),1)
        img = cv2.line(img,corners[3],corners[0],(0,255,255),1)
    
    # display
    cv2.imshow('blank0',img)
    # write frame
    out.write(img)

cap.release()
out.release()
cv2.destroyAllWindows()





