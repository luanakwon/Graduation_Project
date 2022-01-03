import cv2
import numpy as np
import mediapipe as mp


img = cv2.imread('regression_sample.png')

grimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

d_img = cv2.Sobel(grimg,-1,0,1,ksize=3)
d_img = np.abs(d_img)


x = np.arange(d_img.shape[1])
y = np.arange(d_img.shape[0])
xv, yv = np.meshgrid(x,y)

# center high triangle
weight_c = np.abs(yv.astype(np.float32)-d_img.shape[0]/2)
weight_c = np.abs(weight_c - weight_c[0,0])

print('yv')
print(yv[:,:15])

print('weight')
print(weight_c[:,:15])

ptr = 0
left = [0]
right = [d_img.shape[1]]
top = [0]
down = [d_img.shape[0]]
bias = []
search_depth = 4
sd = 2**(search_depth+1)-2 # max number of biases 
while search_depth > 0:
    l,r,t,d = left[ptr],right[ptr],top[ptr],down[ptr]
    ptr += 1

    m = (l+r)//2 # middle
    # left half
    b = np.sum(yv[t:d,l:m]*weight_c[t:d,l:m]*d_img[t:d,l:m])\
        /np.sum(weight_c[t:d,l:m]*d_img[t:d,l:m])
    
    left.append(l)
    right.append(m)
    top.append(max(int(b-(d-t)/4),t))
    down.append(min(int(b+(d-t)/4),d))
    bias.append(b)
    
    # right half
    b = np.sum(yv[t:d,m:r]*weight_c[t:d,m:r]*d_img[t:d,m:r])\
        /np.sum(weight_c[t:d,m:r]*d_img[t:d,m:r])

    left.append(l)
    right.append(m)
    top.append(max(int(b-(d-t)/4),t))
    down.append(min(int(b+(d-t)/4),d))
    bias.append(b)

    if len(bias) >= sd:
        break
print(bias)
print(bias[-2**search_depth:])

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