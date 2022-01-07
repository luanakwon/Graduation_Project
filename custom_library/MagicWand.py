import os
import cv2
import numpy as np

def apply(img, point, tol):
    # image, point, tolerance
    # img of shape h,w,c

    c0 = img[point[0],point[1]]
    intimg = img.copy().astype(np.int32)
    intimg -= c0
    boolimg = (np.max(intimg,axis=2) - np.min(intimg,axis=2)) < tol
    
    return boolimg



# tol = 60

# for filename in os.listdir("cropped_finger"):
#     img = cv2.imread(os.path.join("cropped_finger", filename))
#     cv2.imshow(f'{filename}',img)
#     cv2.waitKey()

#     #grimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     dximg = cv2.Scharr(img,-1,1,0).astype(np.float32)
#     for i in range(3):
#         dximg[:,:,i] -= np.mean(dximg[:,:,i])
#         dximg[:,:,i] = np.abs(dximg[:,:,i])
#         dximg[:,:,i] /= np.max(dximg[:,:,i])
#     dximg = np.max(dximg,axis=2)

#     boolimg = dximg > 0.9
#     cv2.imshow('bool',boolimg.astype(np.float32))
#     cv2.waitKey()

#     height, width = img.shape[:2]
#     c0 = img[height//2,width//2] # pivot color
#     intimg = img.astype(np.int32)
#     intimg -= c0
#     boolimg = (np.max(intimg,axis=2) - np.min(intimg,axis=2)) < tol

#     img[:,:,0] += boolimg.astype(np.uint8)*100
    
#     cv2.imshow('highlight',img)
#     cv2.waitKey()

#     cv2.destroyAllWindows()


