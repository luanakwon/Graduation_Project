import cv2
import numpy as np
import mediapipe as mp


import os
import MagicWand

img = cv2.imread(os.path.join('cropped_finger','0highres.png'))
cv2.imshow('original',img)
height, width = img.shape[:2]

finger_area = MagicWand.apply(img,(height//2,width//2),60).astype(np.int32)
cv2.imshow('mw',finger_area.astype(np.float32))
dx = np.zeros_like(finger_area,dtype=np.int32)
dx[:,1:] = finger_area[:,1:] - finger_area[:,:-1]

x = np.arange(width)
y = np.arange(height)
xv, yv = np.meshgrid(x,y)
xv = xv.astype(np.int32)

# closest F->T pixel from the reference column(0.7 of width)
left_edge_x = np.argmax((xv*dx)[:,:int(width*0.7)],axis=1)
# closest T->F pixel from the reference column(0.3 of width)
right_edge_x = np.argmax(((width-xv)*-dx)[:,int(width*0.3):],axis=1)
right_edge_x += int(width*0.3)
print((xv)[20])


import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot(left_edge_x,label='L')
plt.plot(right_edge_x,label='R')
plt.legend()
plt.show()

# import time
# print(time.time())
# lct = time.localtime()
# print('%4d%02d%02d%02d%02d'%(
#     lct.tm_year,lct.tm_mon,lct.tm_mday,lct.tm_hour,lct.tm_min))

# img = cv2.imread('regression_sample.png')

# grimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# d_img = cv2.Sobel(grimg,-1,0,1,ksize=3)
# d_img = np.abs(d_img)

# print(yv[:,:15])

# print('weight')
# print(weight_c[:,:15])

# ptr = 0
# left = [0]
# right = [d_img.shape[1]]
# top = [0]
# down = [d_img.shape[0]]
# bias = []
# search_depth = 4
# sd = 2**(search_depth+1)-2 # max number of biases 
# while search_depth > 0:
#     l,r,t,d = left[ptr],right[ptr],top[ptr],down[ptr]
#     ptr += 1

#     m = (l+r)//2 # middle
#     # left half
#     b = np.sum(yv[t:d,l:m]*weight_c[t:d,l:m]*d_img[t:d,l:m])\
#         /np.sum(weight_c[t:d,l:m]*d_img[t:d,l:m])
    
#     left.append(l)
#     right.append(m)
#     top.append(max(int(b-(d-t)/4),t))
#     down.append(min(int(b+(d-t)/4),d))
#     bias.append(b)
    
#     # right half
#     b = np.sum(yv[t:d,m:r]*weight_c[t:d,m:r]*d_img[t:d,m:r])\
#         /np.sum(weight_c[t:d,m:r]*d_img[t:d,m:r])

#     left.append(l)
#     right.append(m)
#     top.append(max(int(b-(d-t)/4),t))
#     down.append(min(int(b+(d-t)/4),d))
#     bias.append(b)

#     if len(bias) >= sd:
#         break
# print(bias)
# print(bias[-2**search_depth:])

# # b()() = bias(step)(order from left)
# # d()() = divider(step)(order from left)
# d00 = d_img.shape[1]//2
# b00 = np.sum(yv[:,:d00]*weight_c[:,:d00]*d_img[:,:d00])\
#     /np.sum(weight_c[:,:d00]*d_img[:,:d00])
# b01 = np.sum(yv[:,d00:]*weight_c[:,d00:]*d_img[:,d00:])\
#     /np.sum(weight_c[:,d00:]*d_img[:,d00:])

# print(b00, b01)

# d10 = 1*d_img.shape[1]//4
# s = max(int(b00-d_img.shape[0]/4),0)
# e = min(int(b00+d_img.shape[0]/4),d_img.shape[0])
# b10 = np.sum(yv[s:e,:d10]*weight_c[s:e,:d10]*d_img[s:e,:d10])\
#     /np.sum(weight_c[s:e,:d10]*d_img[s:e,:d10])
# b11 = np.sum(yv[s:e,d10:d00]*weight_c[s:e,d10:d00]*d_img[s:e,d10:d00])\
#     /np.sum(weight_c[s:e,d10:d00]*d_img[s:e,d10:d00])

# d11 = 3*d_img.shape[1]//4
# s = max(int(b01-d_img.shape[0])/4,0)
# e = min(int(b01+d_img.shape[0]/4),d_img.shape[0])
# b12 = np.sum(yv[s:e,d00:d11]*weight_c[s:e,d00:d11]*d_img[s:e,d00:d11])\
#     /np.sum(weight_c[s:e,d00:d11]*d_img[s:e,d00:d11])
# b13 = np.sum(yv[s:e,d11:]*weight_c[s:e,d11:]*d_img[s:e,d11:])\
#     /np.sum(weight_c[s:e,d11:]*d_img[s:e,d11:])

# print(b10,b11,b12,b13)