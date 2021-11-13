import numpy as np
import cv2
import random

def WarpedPerspectiveImage(src,out_size = 200):
    height, width = src.shape[:2]

    c = np.array([[width,height,0,1],
        [-width,height,0,1],
        [-width,-height,0,1],
        [width,-height,0,1]]).transpose()

    thx = random.gauss(0,0.02)
    thy = random.gauss(0,0.02)
    thz = random.gauss(0,0.02)
    if width > height:
        thz += np.pi/2
    tz = random.randint(210,220)
    
    Rx = np.array([[1,0,0,0],
        [0,np.cos(thx),-np.sin(thx),0],
        [0,np.sin(thx),np.cos(thx),0],
        [0,0,0,1]])

    Ry = np.array([[np.cos(thy),0,np.sin(thy),0],
        [0,1,0,0],
        [-np.sin(thy),0,np.cos(thy),tz],
        [0,0,0,1]])

    Rz = np.array([[np.cos(thz),-np.sin(thz),0,0],
        [np.sin(thz),np.cos(thz),0,0],
        [0,0,1,0],
        [0,0,0,1]])


    rot_mat = np.matmul(Rx,Ry)
    rot_mat = np.matmul(rot_mat,Rz)
    toImage = np.array([[1,0,0,0],[0,-1,0,0]])

    xc = np.matmul(rot_mat,c)

    for i in range(4):
        xc[:3,i] /= xc[2,i]

    pts = np.transpose(np.matmul(toImage,xc)*100).astype(np.float32)
    pts -= np.min(pts,axis=0)
    # align center on x axis
    pts[:,0] += (np.max(pts)-np.max(pts,axis=0)[0])//2
    H_out = W_out = np.max(pts).astype(np.int32)
    srcpts = np.array([[width,0],[0,0],[0,height],[width,height]],dtype=np.float32)
    matr = cv2.getPerspectiveTransform(srcpts,pts)
    
    out = cv2.warpPerspective(src,matr,(W_out,H_out))
    out_label = cv2.warpPerspective(np.ones((height, width)),matr,(W_out,H_out))
    
    out = cv2.resize(out,(out_size,out_size))
    out_label = cv2.resize(out_label,(out_size,out_size))
    return (out, out_label)
