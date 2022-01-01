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

def DrawTiltedSquare(src,center,dir,size,color,thickness=1):
    dv = dir.astype(np.float32).copy()
    dv /= np.sqrt(np.sum(dv*dv))
    dv = (dv*size).astype(np.int32)
    Hdv = np.flip(dv).copy()
    Hdv[1] *= -1
    start = center - dv//2 - Hdv//2
    dst = cv2.line(src,start,start+dv,color,thickness)
    dst = cv2.line(dst,start+dv,start+dv+Hdv,color,thickness)
    dst = cv2.line(dst,start+dv+Hdv,start+Hdv,color,thickness)
    dst = cv2.line(dst,start+Hdv,start,color,thickness)
    return dst

def CropTiltedSquare(src,center,dir,size,dsize):
    dv = dir.astype(np.float32).copy()
    dv /= np.sqrt(np.sum(dv*dv))
    dv = (dv*size).astype(np.int32)
    Hdv = np.flip(dv).copy()
    Hdv[1] *= -1
    start = center - dv//2 - Hdv//2
    s = dsize
    input_pts = np.float32([start,start+dv,start+dv+Hdv,start+Hdv]).reshape(4,2)
    output_pts = np.float32([[0,0],[s-1,0],[s-1,s-1],[0,s-1]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    dst = cv2.warpPerspective(src,M,(s,s),flags=cv2.INTER_LINEAR)
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
    area_diff_thres = 0.2 # heuristic value
    frames_to_use = np.zeros((100,img_size[1],img_size[0],3),dtype=np.uint8)
    corners_to_use = np.zeros((100,4,2),dtype=np.float32)
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
        original_img = cv2.flip(frame,flipCode=1)

        # change to portrait of ratio 800:600
        original_img = cv2.resize(original_img[:,140:500],dsize=img_size)
        
        img = original_img.copy()

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
            frames_to_use[frame_counter] = original_img
            rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            rgb_img.flags.writeable = False
            ret,dip,dir = FDDtector.run(rgb_img)
            FDD_rets[frame_counter] = ret
            FDD_dips[frame_counter] = dip
            FDD_dirs[frame_counter] = dir
            corners_to_use[frame_counter] = np.float32(corners).reshape((4,2))

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
    frame_counter = 0
    for i in range(frames_to_use.shape[0]):
        frame = frames_to_use[i].copy()
        if FDD_rets[i]:
            for dip, dir in zip(FDD_dips[i],FDD_dirs[i]):
                frame = DrawSquare(frame,dip,100,(0,0,255),1)
                frame = cv2.line(frame,dip,dip+dir.astype(np.int32),(255,0,0),1)
                frame_counter = i
        cv2.imshow("frames_to_use",frame)
        cv2.waitKey(66)

    print(frame_counter)
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

    # seems like mediapipe hand detection doesn't work very well
    # with palm covered. Since it tends to shrink the hands, choose
    # the frame with finger dips spreaded wide
    max_wide = 0
    for i in range(frames_to_use.shape[0]):
        if FDD_rets[i]:
            wideness = (-FDD_dips[i][0][0]-FDD_dips[i][1][0]
                +FDD_dips[i][2][0]+FDD_dips[i][3][0])
            if max_wide < wideness:
                frame_counter = i
                max_wide = wideness

    # warp perspective
    img = frames_to_use[frame_counter]
    input_pts = corners_to_use[frame_counter]
    output_pts = np.array(
        [crop_window[0][1],crop_window[0][0],
        crop_window[1][1],crop_window[0][0],
        crop_window[1][1],crop_window[1][0],
        crop_window[0][1],crop_window[1][0]],dtype=np.float32).reshape((4,2))
    print(input_pts, output_pts)
    M=cv2.getPerspectiveTransform(input_pts,output_pts)
    T_img = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    # transfrom 8 FDD points
    FDD_cors = np.ones((3,8))
    FDD_cors[:2,:4] = np.transpose(FDD_dips[frame_counter])
    FDD_cors[:2,4:] = np.transpose(FDD_dirs[frame_counter])
    FDD_cors = np.matmul(M,FDD_cors)
    FDD_cors = (FDD_cors/FDD_cors[2]).astype(np.int32)
    FDD_cors = np.transpose(FDD_cors[:2])

    # crop and save 4 finger dips
    ROI_crops = np.zeros((4,200,200,3),dtype=np.uint8)
    for i, (dip, dir) in enumerate(zip(FDD_cors[:4],FDD_cors[4:])):
        ROI_crops[i] = CropTiltedSquare(T_img,dip,dir,100,200)

    # draw indicators
    for dip, dir in zip(FDD_cors[:4],FDD_cors[4:]):
        T_img = DrawTiltedSquare(T_img,dip,dir,100,(0,255,0),1)
        T_img = cv2.line(T_img,dip,dip+dir,(255,255,0),1)


    cv2.imshow('Transfrormed img',T_img)
    cv2.waitKey()
    for i, crop in enumerate(ROI_crops):
        cv2.imshow(f'crop{i}',crop)
        cv2.waitKey()
    cv2.imwrite('shba_hand.png',T_img)


    
if __name__ == "__main__":
    WebcamDemo()
    