import numpy as np
import cv2

def linear_regression2(weighted_img, threshold):
    # my simple linear regression method
    height, width = weighted_img.shape[:2]
    filter = weighted_img > threshold

    x = np.arange(weighted_img.shape[1])
    y = np.arange(weighted_img.shape[0])
    xv, yv = np.meshgrid(x,y)
    
    xv = (xv*filter)[filter > 0]
    yv = (yv*filter)[filter > 0]

    lb = np.repeat(np.arange(height),height)
    rb = np.tile(np.arange(height),height)

    sq_d = np.matmul(xv.reshape(-1,1),(rb-lb).reshape(1,-1))/width
    sq_d = (sq_d + lb - yv.reshape(-1,1))**2

    scores = np.sum((sq_d<5).astype(np.int32) + (sq_d<2.5).astype(np.int32), axis=0)
    # scores = np.sum(1/(0.1+sq_d), axis=0)
    idx = np.argmax(scores)
    if scores[idx] < 0.8*2*width:
        return 0, 0
    else:
        return lb[idx], rb[idx]

def get_cw(img_size,p_c,p_w):
    # get coordinate required for cropping windows
    # img_size of order W,H
    card_size = np.array([[0,0],[8560,5398]],dtype=np.float64)
    if img_size[0]/img_size[1] > 0.63: # fit to height
        card_size *= p_c*img_size[1]/card_size[1][0]
    else: # fit to width
        card_size *= p_c*img_size[0]/card_size[1][1]
    
    dw = int(card_size[1][1]*p_w*0.5)
    card_size[:,0] += (img_size[1]-card_size[1][0])/2
    card_size[:,1] += (img_size[0]-card_size[1][1])/2
    card_size = card_size.astype(np.int32)
    
    cw = [
    [card_size[0][0]-dw,card_size[0][0]+dw,card_size[0][1],card_size[1][1]],
    [card_size[1][0]-dw,card_size[1][0]+dw,card_size[0][1],card_size[1][1]],
    [card_size[0][0],card_size[1][0],card_size[0][1]-dw,card_size[0][1]+dw],
    [card_size[0][0],card_size[1][0],card_size[1][1]-dw,card_size[1][1]+dw]]

    return card_size, cw

def run(grimg, p_c, p_w):
    # main method that returns coordinates of 4 corners
    crop_window, cw = get_cw((grimg.shape[1],grimg.shape[0]),p_c,p_w)

    possible_edges = []
    possible_edges.append(grimg[cw[0][0]:cw[0][1],cw[0][2]:cw[0][3]])
    possible_edges.append(grimg[cw[1][0]:cw[1][1],cw[1][2]:cw[1][3]])
    possible_edges.append(
        cv2.rotate(grimg[cw[2][0]:cw[2][1],cw[2][2]:cw[2][3]],cv2.ROTATE_90_CLOCKWISE))
    possible_edges.append(
        cv2.rotate(grimg[cw[3][0]:cw[3][1],cw[3][2]:cw[3][3]],cv2.ROTATE_90_CLOCKWISE))

    pt8 = []

    for i, pedge in enumerate(possible_edges):
        pedge = pedge.astype(np.float64)
        pedge -= np.min(pedge)
        pedge /= np.max(pedge)

        dysq = cv2.Sobel(pedge,-1,0,1,ksize=5)
        dysq = dysq**2

        std_thres = 0.3
        thres = std_thres*np.std(dysq) + np.mean(dysq)
        left_bias, right_bias = linear_regression2(dysq,thres)
        
        if left_bias != 0 or right_bias != 0:
            if i < 2:
                pt1 = [cw[i][2],cw[i][0]+left_bias]
                pt2 = [cw[i][3]-1,cw[i][0]+right_bias]
            else:
                pt1 = [cw[i][2]+left_bias,cw[i][1]-1]
                pt2 = [cw[i][2]+right_bias,cw[i][0]]
        else:
            pt1 = [-1,-1]
            pt2 = [-1,-1]
        pt8.append(pt1)
        pt8.append(pt2)
    
    pt8 = np.array(pt8).reshape(4,4)
    
    A = np.zeros((8,2))
    C = np.zeros((8,1))
    corners = []
    points_found = not (pt8 < 0).any()
    if points_found:
        for i, p in enumerate(pt8):
            C[i*2] = p[0]*p[3] - p[2]*p[1]
            p = p[2:] - p[:2]
            A[i*2,0] = p[1]
            A[i*2,1] = -p[0]

        A[1] = A[4]
        A[3] = A[6]
        A[5] = A[2]
        A[7] = A[0]

        C[1] = C[4]
        C[3] = C[6]
        C[5] = C[2]
        C[7] = C[0]
            
        corners.append(np.matmul(np.linalg.inv(A[0:2]),C[0:2]).flatten().astype(np.int32))
        corners.append(np.matmul(np.linalg.inv(A[6:8]),C[6:8]).flatten().astype(np.int32))
        corners.append(np.matmul(np.linalg.inv(A[2:4]),C[2:4]).flatten().astype(np.int32))
        corners.append(np.matmul(np.linalg.inv(A[4:6]),C[4:6]).flatten().astype(np.int32))

    return points_found, corners
        

    