import cv2
import numpy as np
import matplotlib.pyplot as plt
import CardDetection
import FingerDipDetection

def DrawSquare(src,center, size, color, thickness=1):
    cx,cy = center
    s = size//2
    dst = cv2.line(src,(cx-s,cy-s),(cx+s,cy-s),color,thickness)
    dst = cv2.line(dst,(cx+s,cy-s),(cx+s,cy+s),color,thickness)
    dst = cv2.line(dst,(cx+s,cy+s),(cx-s,cy+s),color,thickness)
    dst = cv2.line(dst,(cx-s,cy+s),(cx-s,cy-s),color,thickness)
    return dst

def WebcamDemo():
    # CardDetection settings
    img_size = (480,640) # W,H
    p_c = 0.4 # size of gray grid line in length percent
    p_w = 0.2 # size of edge detection window, ratio wrt grid center cell 

    # FingerDipDetection settings
    FDDtector = FingerDipDetection.FingerDipDetector()

    # choose 100 continuous steady frames
    area1 = 0 # pseudo area of current frame
    area0 = 0 # pseudo area of last fram
    area_base = img_size[0]*img_size[1]*(p_c**2) # area of center grid cell(for norm)
    frame_counter = 0
    area_diff_thres = 0.1 # heuristic value
    frames_to_use = np.zeros((100,img_size[1],img_size[0],3),dtype=np.uint8)
    corners_to_use = np.zeros((100,4,2))
    FDD_rets = np.zeros((100),dtype=np.uint8)
    FDD_dips = np.zeros((100,4,2),dtype=np.int32)
    FDD_dirs = np.zeros((100,4,2))

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
            area1 /= area_base
            
        if not ret or abs(area1-area0) > area_diff_thres:
            frame_counter = -1
        else:
            frames_to_use[frame_counter] = img
            rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            rgb_img.flags.writeable = False
            ret,dip,dir = FDDtector.run(rgb_img)
            FDD_rets[frame_counter] = ret
            FDD_dips[frame_counter] = dip
            FDD_dirs[frame_counter] = dir

        area0 = area1
        
        # display
        cv2.imshow('blank0',img)
        #temp
        frame_counter += 1
        if frame_counter >= frames_to_use.shape[0]:
            break
        elif frame_counter == 10:
            print(10)
        elif frame_counter == 50:
            print(50)
        elif frame_counter == 95:
            print(95)
    
    # display frames to use
    print("presskey")
    cv2.waitKey()
    for i in range(frames_to_use.shape[0]):
        frame = frames_to_use[i]
        if FDD_rets[i]:
            for dip, dir in zip(FDD_dips[i],FDD_dirs[i]):
                frame = DrawSquare(frame,dip,100,(0,0,255),1)
                frame = cv2.line(frame,dip,dip+dir.astype(np.int32),(255,0,0),1)
        cv2.imshow("frames_to_use",frame)
        cv2.waitKey(66)

    # save frames to use as mp4 video
    vid_name = input("save video as (type q to quit)")
    if vid_name != 'q':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vid_name,fourcc,15.0,img_size)
        for frame in frames_to_use:
            out.write(frame)
        out.release()

    # choose one frame from frames to use
    # I think we'll use 1 frame instead of 100
    # we will finely adjust the corners,
    # but not implemented in this demo
    img = frames_to_use[-1]
    input_pts = np.array(corners,dtype=np.float32).reshape((4,2))
    output_pts = np.array(
        [crop_window[0][1],crop_window[0][0],
        crop_window[1][1],crop_window[0][0],
        crop_window[1][1],crop_window[1][0],
        crop_window[0][1],crop_window[1][0]],dtype=np.float32).reshape((4,2))
    print(input_pts, output_pts)
    M=cv2.getPerspectiveTransform(input_pts,output_pts)
    T_img = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

    cv2.imshow('Transfrormed img',T_img)
    cv2.waitKey()
    cv2.imwrite('shba_hand.png',T_img)


    
if __name__ == "__main__":
    WebcamDemo()
    