import numpy as np
import cv2
import CardDetection

img_size = (480,640) # W,H
p_c = 0.4 # center grid percent
p_w = 0.2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

crop_window, _ = CardDetection.get_cw(img_size,p_c,p_w)

while cv2.waitKey(33) < 0:
    ret, img = cap.read()
    img = cv2.flip(img,flipCode=1)
    img = cv2.resize(img[:,140:500],dsize=img_size)

    grimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img[crop_window[0][0],:] = (150,150,150)
    img[crop_window[1][0],:] = (150,150,150)
    img[:,crop_window[0][1]] = (150,150,150)
    img[:,crop_window[1][1]] = (150,150,150)

    ret, corners = CardDetection.run(grimg,p_c,p_w)
    
    if ret:
        print(f'{ret}, {corners}')
        img = cv2.line(img,corners[0],corners[1],(0,255,255),1)
        img = cv2.line(img,corners[1],corners[2],(0,255,255),1)
        img = cv2.line(img,corners[2],corners[3],(0,255,255),1)
        img = cv2.line(img,corners[3],corners[0],(0,255,255),1)
    

    cv2.imshow('blank0',img)

cap.release()
cv2.destroyAllWindows()





