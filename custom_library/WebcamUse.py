import cv2
import numpy as np
import matplotlib.pyplot as plt
import CardDetection

def WebcamDemo():
    #temp
    frame_counter = 0
    #list_area_diff = np.zeros((300))
    area1 = 0
    area0 = 0
    area_diff_thres = 0.1

    # CardDetection settings
    img_size = (480,640) # W,H
    p_c = 0.4 # size of gray grid line in length percent
    p_w = 0.2 # size of edge detection window, ratio wrt grid center cell 

    # cordinate required to draw gray grid line
    crop_window, _ = CardDetection.get_cw(img_size,p_c,p_w)

    # turn on the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("webcam window width and height")
    print(cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)

    

    # framerate at about 30fps.
    # can get slower due to other operations
    while cv2.waitKey(33) < 0:
        # read image from webcam
        ret, frame = cap.read()
        if not ret:
            print("frame unavailable")
            break

        # mirror the frame
        img = cv2.flip(frame,flipCode=1)

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
        area1 = 0
        # draw card contour
        if ret:
            img = cv2.line(img,corners[0],corners[1],(0,255,255),1)
            img = cv2.line(img,corners[1],corners[2],(0,255,255),1)
            img = cv2.line(img,corners[2],corners[3],(0,255,255),1)
            img = cv2.line(img,corners[3],corners[0],(0,255,255),1)
            
            for c in corners:
                area1 += abs(c[0]*c[1])
            #normalize detected area with the area of center grid cell
            area1 /= img_size[0]*img_size[1]*(p_c**2)
            
        if not ret or abs(area1-area0) > area_diff_thres:
            frame_counter = 66

        area0 = area1
        
        # display
        cv2.imshow('blank0',img)
        #temp
        frame_counter -= 1
        if frame_counter <= 0:
            print(frame_counter)
            break
        elif frame_counter <= 30:
            print(frame_counter)
        elif frame_counter < 60:
            print(frame_counter)
    
    print("presskey")
    cv2.waitKey()

    

if __name__ == "__main__":
    WebcamDemo()
    