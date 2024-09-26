import numpy as np
import cv2

a = """
I/System.out: 250 25 128 148 132 144 137 135 122 129 
    138 142 130 148 130 143 139 136 122 132 
I/System.out: 140 141 135 147 130 143 140 134 125 134 
    138 142 137 146 130 143 140 132 127 132 
    136 140 138 146 129 143 140 131 126 129 
    137 139 139 147 129 141 141 132 124 126 
    138 140 140 150 130 140 142 134 125 124 
    138 140 141 151 130 140 142 132 126 122 
    136 142 141 148 131 140 142 130 125 121 
I/System.out: 136 141 139 147 131 141 143 132 125 122 
    137 141 139 147 131 142 143 136 123 121 
    137 140 139 147 130 143 143 138 122 119 
    138 139 140 147 130 142 143 138 123 120 
    139 139 141 146 131 142 143 137 123 121 
    140 140 140 145 131 142 143 136 123 121 
    142 140 140 143 130 142 142 136 124 121 
I/System.out: 142 139 140 142 128 143 143 136 125 121 
    142 139 140 142 128 141 142 135 124 121 
    141 139 141 142 130 140 142 135 123 121 
    139 138 140 141 131 140 142 137 123 121 
    139 136 138 138 131 140 141 137 122 122 
    139 132 138 136 131 140 141 138 121 122 
    138 130 138 136 131 141 141 139 121 123 
I/System.out: 137 130 138 136 132 142 140 139 123 123 
    137 131 138 135 133 144 141 139 123 123 
    136 132 138 134 134 143 141 139 123 124 
    135 132 137 132 134 144 141 139 124 125 
    134 132 138 131 135 145 141 139 124 124 
    135 132 138 128 135 145 140 138 122 124 
    135 130 137 127 134 145 140 138 121 124 
I/System.out: 136 129 136 125 133 144 143 138 121 123 
    135 127 137 125 132 143 142 139 121 121 
    132 126 138 126 130 145 141 139 123 118 
    132 126 137 128 130 144 141 139 124 119 
    133 126 138 130 130 143 142 139 124 124 
    136 126 137 132 131 143 142 139 126 127 
    137 126 136 134 130 143 142 137 125 125 
I/System.out: 137 126 136 134 129 144 142 137 123 124 
    135 125 136 136 129 144 143 137 123 125 
    134 124 135 136 130 144 143 137 123 126 
    135 125 133 137 132 144 142 138 121 127 
    136 126 133 137 134 143 142 137 121 125 
    138 128 131 137 134 143 142 138 122 125 
I/System.out: 137 128 130 137 133 145 141 138 125 126 
    133 128 130 137 132 144 141 137 126 126 
    133 127 130 139 131 142 141 137 126 127 
    133 127 131 141 131 143 142 137 126 128 
    133 128 131 141 131 145 142 137 125 128 
    135 130 130 141 132 144 142 137 127 127 
    135 130 130 141 132 143 145 137 127 126 
I/System.out: 134 130 129 141 132 143 145 137 127 126 
    132 131 130 142 131 144 145 137 127 125 
    132 133 131 143 130 144 146 138 126 125 
    134 134 131 143 130 143 146 138 126 125 
    135 135 131 143 131 142 145 136 128 124 
    135 137 131 143 133 141 144 136 129 124 
    134 139 130 143 133 141 143 140 128 126 
I/System.out: 135 139 131 144 132 139 143 140 126 127 
    137 140 136 145 132 139 142 141 128 128 
    137 140 139 145 133 139 141 141 130 127 
    136 140 137 146 134 138 140 140 136 127 
    135 141 136 147 135 137 139 140 135 129 
    134 141 134 146 136 137 139 140 130 129 
    134 140 133 145 136 138 139 140 130 127 
I/System.out: 136 140 131 146 134 138 138 138 130 127 
    137 141 131 147 133 138 137 138 131 126 
    138 142 134 146 134 139 138 138 132 125 
    137 142 136 145 135 139 137 137 132 125 
    138 142 136 145 134 139 135 136 133 124 
    140 142 136 145 134 138 136 136 134 126 
I/System.out: 141 143 137 145 137 138 135 136 135 129 
    139 142 136 147 138 137 133 135 136 129 
    137 143 136 148 136 136 134 133 136 132 
    138 143 135 148 135 137 134 132 136 134 
    139 144 135 148 137 138 133 132 135 131 
    138 144 135 147 139 138 133 131 136 129 
    136 143 136 147 139 138 133 129 136 129 
I/System.out: 137 142 135 147 139 137 133 130 136 129 
    138 142 133 146 138 137 132 130 136 128 
    138 144 132 146 138 137 132 129 136 127 
    137 145 133 146 137 136 132 129 136 126 
    138 146 132 146 137 136 131 129 137 126 
    138 146 133 146 137 136 132 127 137 126 
I/System.out: 136 146 134 146 138 137 132 126 137 127 
    135 146 135 147 139 136 131 127 136 128 
    134 146 137 146 139 137 129 126 135 131 
    134 146 137 145 138 136 128 126 135 132 
    134 146 136 146 137 136 127 126 136 132 
I/System.out: 134 145 137 146 138 136 126 126 137 132 
    135 143 137 145 137 134 125 127 135 133 
    135 142 137 145 138 135 125 126 135 134 
    136 141 138 144 137 134 123 123 136 134 
    135 141 139 143 137 131 122 122 135 133 
    132 141 139 144 135 129 121 120 136 135 
    130 141 138 145 134 130 124 129 141 140 
I/System.out: 132 143 141 146 140 137 138 143 144 143 
    141 148 147 148 146 144 145 145 143 144 
    150 152 150 148 147 146 145 145 143 145 
    152 152 151 150 149 148 146 145 144 146 
    153 153 152 150 150 149 147 146 145 146 
    152 153 151 149 150 149 147 145 144 144 
I/System.out: 153 152 150 149 149 148 147 144 145 144 
    152 152 151 149 150 147 147 144 145 145 
    151 151 151 148 150 148 147 144 145 145 
    150 150 151 148 151 147 146 143 144 145 
    151 151 150 148 150 147 146 144 144 145 
    152 152 150 149 149 148 147 143 144 144 
I/System.out: 152 152 150 150 149 149 148 144 144 145 
    152 152 149 149 150 148 147 145 144 146 
    152 152 150 149 150 147 147 146 143 145 
    152 152 150 149 150 147 147 146 143 144 
    151 151 148 149 150 147 146 146 143 143 
    150 150 149 149 149 147 145 145 144 143 
I/System.out: 150 150 150 149 149 147 146 146 145 144 
    151 150 149 149 149 146 146 146 145 144 
    151 150 148 149 149 148 146 145 144 144 
    150 150 148 149 149 148 147 144 145 145 
    150 151 149 149 148 145 147 145 145 145 
    150 151 149 149 144 137 147 147 144 144 
    149 151 150 150 138 128 147 147 145 144 
I/System.out: 149 150 150 149 131 126 147 146 144 145 
    150 151 149 150 129 125 147 146 144 145 
    151 151 149 151 126 124 146 146 145 145 
    150 152 149 148 126 125 145 146 145 145 
    149 151 150 145 126 127 143 147 145 144 
    149 149 150 142 126 125 137 147 143 144 
I/System.out: 149 148 149 136 127 125 129 146 143 144 
    149 150 148 129 126 128 124 146 144 144 
    149 151 148 127 124 126 124 145 145 143 
    149 150 148 126 124 123 125 146 145 143 
    150 149 149 125 124 122 125 146 145 144 
I/System.out: 150 150 149 124 123 123 124 146 145 144 
    150 151 148 125 123 124 123 146 145 144 
    151 151 148 125 122 123 122 145 144 145 
    151 151 148 124 123 123 122 145 144 145 
    151 150 147 125 123 123 122 144 144 145 
    151 150 147 127 123 124 122 144 144 144 
    150 150 147 127 123 123 122 143 143 144 
I/System.out: 150 150 148 128 124 121 123 144 144 145 
    150 150 148 128 124 119 123 144 145 144 
    150 149 148 125 125 119 122 142 145 145 
    150 149 148 120 124 117 116 138 144 145 
    149 149 148 117 123 117 125 133 145 144 
    149 149 147 122 124 145 167 129 145 145 
I/System.out: 149 149 147 130 125 163 184 125 145 144 
    149 149 146 121 122 160 179 121 145 144 
    150 149 147 103 117 163 179 120 145 144 
    150 148 149 100 112 164 179 120 144 145 
    149 148 147 115 104 164 178 121 144 145 
    149 148 139 120 101 163 178 120 145 144 
I/System.out: 149 148 126 119 101 163 177 121 145 143 
I/System.out: 149 148 128 129 100 161 175 122 144 144 
I/System.out: 150 149 129 124 102 160 174 122 144 145 
    150 148 112 99 102 161 173 122 143 145 
    150 149 96 92 101 160 172 121 143 145 
    150 148 109 109 98 160 171 120 143 143 
I/System.out: 149 147 127 127 96 160 172 119 144 142 
    149 148 129 126 97 162 174 120 145 144 
    148 148 130 128 98 139 175 120 146 145 
    147 149 130 124 100 96 169 121 145 145 
    147 148 125 115 105 124 134 121 144 145 
    149 149 126 112 105 173 118 121 144 145 
I/System.out: 149 149 137 122 103 169 163 121 144 144 
    149 148 138 142 101 168 177 120 143 143 
    148 147 122 146 101 170 168 120 142 142 
    148 147 109 133 102 170 170 122 143 141 
I/System.out: 148 147 123 136 100 171 170 122 144 141 
    149 148 124 136 98 171 171 122 144 142 
    149 148 128 137 93 171 170 123 142 143 
I/System.out: 150 150 134 137 91 171 168 122 144 143 
    151 150 146 136 93 171 170 121 144 144 
    151 148 154 133 94 171 171 122 143 142 
    151 147 143 118 94 171 172 122 142 142 
    150 148 133 97 94 172 173 122 142 142 
    149 149 115 88 98 172 174 121 141 141 
I/System.out: 150 149 92 95 103 171 176 120 139 141 
    151 149 118 115 102 171 177 120 137 141 
    151 149 143 126 99 171 178 120 134 141 
    152 149 142 120 99 170 175 121 131 141 
    152 148 134 113 100 173 175 121 129 141 
I/System.out: 152 149 137 131 99 139 178 121 127 141 
    151 150 162 153 96 101 182 121 124 141 
"""
a = a.replace('I/System.out: ',' ')
b = [int(x) for x in a.split()]
c =  np.array(b)
c = c*256/np.max(c)
c = c.reshape(182,10)
c = c.astype(np.uint8)
d = cv2.resize(c, (0,0), fx=3*16, fy=3)
'(732,115)'
cv2.imshow('t',d)
cv2.waitKey()




# import cv2
# import numpy as np
# import mediapipe as mp
# import CardDetection2

# height, width = (6,5)

# x = np.arange(width)
# y = np.arange(height)
# xv, yv = np.meshgrid(x,y)
# print(xv)
# print(yv)
# print(xv.reshape(-1,1))

# lb = np.repeat(np.arange(height),height)
# rb = np.tile(np.arange(height),height)
# print(lb)
# print(rb)
# sq_d = np.matmul(xv.reshape(-1,1),(rb-lb).reshape(1,-1))/width
# sq_d = np.abs(sq_d + lb - yv.reshape(-1,1))
# for i in sq_d:
#     for j in i:
#         print('%5.1f'%(j),end='')
#     print()


# blank = np.zeros((640,360,4),dtype=np.uint8)
# blank = cv2.resize(blank,(640,360))
# CDDtector = CardDetection2.CardDetector((170,450),(160,400),0.2)

# cgp = CDDtector.getCardGuidePoint(order='xy')
# cgp = cgp.astype(np.int32)

# print(cgp)

# img = cv2.line(blank,cgp[0],cgp[1],(0,225,249,255),1)
# img = cv2.line(img,cgp[1],cgp[2],(0,225,249,255),1)
# img = cv2.line(img,cgp[2],cgp[3],(0,225,249,255),1)
# img = cv2.line(img,cgp[3],cgp[0],(0,225,249,255),1)

# cv2.imshow('blank',img)
# cv2.waitKey()
# cv2.imwrite('CardGuide_BGRA.png',img)
# cv2.imwrite('CardGuide_flip_BGRA.png',cv2.flip(img,1))
# import os

# from numpy.core.fromnumeric import argmax, argmin
# import MagicWand

# img = cv2.imread(os.path.join('cropped_finger','0highres.png'))
# cv2.imshow('original',img)
# height, width = img.shape[:2]

# finger_area = MagicWand.apply(img,(height//2,width//2),60).astype(np.int32)
# cv2.imshow('mw',finger_area.astype(np.float32))
# dx = np.zeros_like(finger_area,dtype=np.int32)
# dx[:,1:] = finger_area[:,1:] - finger_area[:,:-1]

# x = np.arange(width)
# y = np.arange(height)
# xv, yv = np.meshgrid(x,y)
# xv = xv.astype(np.int32)

# # closest F->T pixel from the reference column(0.7 of width)
# left_edge_x = np.argmax((xv*dx)[:,:int(width*0.7)],axis=1)
# # closest T->F pixel from the reference column(0.3 of width)
# right_edge_x = np.argmax(((width-xv)*-dx)[:,int(width*0.3):],axis=1)
# right_edge_x += int(width*0.3)
# print((xv)[20])

# distances = np.zeros_like(left_edge_x)
# points = []

# # algorithm that goes back and forth between left and right edge
# edge0 = left_edge_x
# edge1 = right_edge_x
# p0_y = 0
# p1_y = 0
# while True:
#     points.append((edge0[p0_y],p0_y))

#     p0_x = edge0[p0_y]
#     d = []
#     for i in range(2):
#         if p1_y+i < len(edge1):
#             d.append(np.sqrt((p1_y+i-p0_y)**2+(p0_x-edge1[p1_y+i])**2))
#     if len(d) == 0:
#         print(not d)
#         break

#     print(argmin(d),min(d))
#     distances[p0_y] = min(d)

#     t = edge0
#     edge0 = edge1
#     edge1 = t

#     t = p0_y+1
#     p0_y = p1_y+argmin(d)
#     p1_y = t

# img2 = img.copy()
# for p0, p1 in zip(points[:-1],points[1:]):
#     cv2.line(img2,p0,p1,(0,100,100),1)

# cv2.imshow('shoelace',img2)

# #smoothing the outliers(distance == 0)
# for i, d in enumerate(distances):
#     if d == 0:
#         # for the first value
#         if i == 0:
#             for d_next in distances[i+1:]:
#                 if d_next != 0:
#                     d = d_next
#         # for the last value
#         elif i == len(distances)-1:
#             for d_last in distances[:i].flipud():
#                 if d_last != 0:
#                     d = d_last
#         # for the middle value
#         else:
#             for j in range(min([i+1,len(distances)-i,10])):
#                 d_last = distances[i-j]
#                 d_next = distances[i+j]
#                 if d_last != 0 and d_next != 0:
#                     d = (d_last+d_next)/2

#         if d == 0:
#             raise ValueError('former process was bullshit')
#         else:
#             distances[i] = d
        
                

# import matplotlib.pyplot as plt
# plt.figure(figsize=(5,5))
# plt.plot(left_edge_x,label='L')
# plt.plot(right_edge_x,label='R')
# plt.plot(distances,label='D')
# plt.legend()
# plt.show()

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