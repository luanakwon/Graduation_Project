import numpy as np
import cv2
import time

kernel_size = 21
seek_window = 13
stride = 3

if (seek_window-1)%stride != 0:
    raise ValueError('wrong stride : (seek_window-1)mod(stride) should be 0')

size_strided_window = ((seek_window-1)//stride + 1)

x = np.arange(50,500,10,dtype=np.int32)
y = np.arange(900,50,-10,dtype=np.int32)
xv,yv = np.meshgrid(x,y)
grid = np.stack((yv,xv),axis=-1)


# input video of size 960*540
cap = cv2.VideoCapture('samplevideo.mp4')
# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sampletracking.mp4',fourcc,30.0,(540,960))

progress = 0

ret, frame0 = cap.read()
frame0 = cv2.resize(frame0,dsize=(540,960))
frame0 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):

    fr = np.zeros((960,540))
    for g in grid:
        for p in g:
            fr[p[0],p[1]] = 1
    cvtfr = cv2.cvtColor((fr*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    out.write(cvtfr)
    progress+=1
    print(f"\rprogress : {progress} ",end='')

    ret, frame1 = cap.read()
    if not ret:
        break
    frame1 = cv2.resize(frame1,dsize=(540,960))
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    offset = seek_window//2
    H = frame1.shape[0] + offset*2
    W = frame1.shape[1] + offset*2
    padded = np.zeros((H,W),dtype=np.float32)
    padded[offset:H-offset,offset:W-offset] += frame1.astype(np.float32)/255

    kernel = np.zeros_like(padded)
    kernel = np.repeat(np.expand_dims(kernel,0),size_strided_window**2,axis=0)

    for i in range(0,seek_window,stride):
        for j in range(0,seek_window,stride):
            kernel[i//stride*size_strided_window+j//stride,i:H-2*offset+i,j:W-2*offset+j] \
                    += frame0.astype(np.float32)/255

    kernel = (kernel-padded)**2

    C,H = kernel.shape[:2]
    kernel = np.reshape(kernel,(C*H,-1))
    diff = cv2.filter2D(kernel,-1,np.ones((kernel_size,kernel_size)))
    diff = np.reshape(diff,(C,H,-1))

    # print(np.mean(np.min(diff,axis=0)),np.max(np.min(diff, axis=0)),end=' ')
    
    t1 = time.time()
    diff[C//2] -= ((np.mean(diff,axis=0) - np.min(diff,axis=0)) < 5)*1000
    #diff[C//2] *= (np.min(diff+(diff == np.min(diff,axis=0))*1000,axis=0) <= move_threshold)
    print(time.time() - t1, end='      ')
    diff = np.argmin(diff,axis=0)

    for g in grid:
        for p in g:
            dh = diff[p[0],p[1]]//size_strided_window*stride - seek_window//2
            dw = int(diff[p[0],p[1]])%size_strided_window*stride - seek_window//2
            p[0] += dh
            p[1] += dw
    grid[:,:,0] = np.clip(grid[:,:,0],0,959)
    grid[:,:,1] = np.clip(grid[:,:,1],0,539)

    frame0 = frame1.copy()

cap.release()
out.release()

