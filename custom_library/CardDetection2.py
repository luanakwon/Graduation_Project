import cv2
import numpy as np

class CardDetector:
    def __init__(self, center, short_p, p_w):
        self.p_w = p_w
        center = np.int32(center)
        short_p = np.int32(short_p)

        short_v = ((short_p-center)*2).astype(np.float32)
        long_v = np.flip(short_v)*np.float32([1,-1])*1.5857
        start_p = short_p-(long_v/2)

        self.input_pts = np.float32([
            start_p,start_p+long_v,start_p-short_v+long_v,start_p-short_v])
        
        self.area = np.linalg.norm(short_v)*np.linalg.norm(long_v)

        short_v *= (1+p_w)
        long_v *= (1+p_w)
        start_p = center + short_v/2 - long_v/2

        self.offset_in_pts = np.float32([
            start_p,start_p+long_v,start_p-short_v+long_v,start_p-short_v])
        self.offset_in_pts = np.flip(self.offset_in_pts,axis=1)
        
        self.short_d = int(np.linalg.norm(short_v))
        self.long_d = int(np.linalg.norm(long_v))

        self.offset_out_pts = np.float32([
            [0,0],[0,self.long_d],[self.short_d,self.long_d],[self.short_d,0]])
        self.offset_out_pts = np.flip(self.offset_out_pts,axis=1)

        self.M_in2out = cv2.getPerspectiveTransform(
            self.offset_in_pts,self.offset_out_pts)

        self.M_out2in = cv2.getPerspectiveTransform(
            self.offset_out_pts,self.offset_in_pts)

    def getCardGuidePoint(self, order='xy'):
        if order == 'xy':
            return np.flip(self.input_pts,axis=1)
        else:
            return self.input_pts

    def getCardGuideArea(self):
        return self.area

    def linear_regression2(self,weighted_img, threshold):
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

    def getPossibleEdge(self, dst):
        ld = int(self.long_d * (self.p_w/(self.p_w+1)))
        ld2 = self.long_d-ld
        sd = int(self.short_d * (self.p_w/(self.p_w+1)))
        sd2 = self.short_d-sd

        # order : top, bottom, left, right
        possible_edges = []
        possible_edges.append(dst[0:sd,ld:ld2])
        possible_edges.append(dst[sd2:self.short_d,ld:ld2])
        possible_edges.append(
            cv2.rotate(dst[sd:sd2,0:ld],
            cv2.ROTATE_90_CLOCKWISE))
        possible_edges.append(
            cv2.rotate(dst[sd:sd2,ld2:self.long_d],
            cv2.ROTATE_90_CLOCKWISE))


        return possible_edges, (ld,ld2,sd,sd2)


    def run(self, grimg):
        src = grimg.copy()
        # orientation : landscape
        dst = cv2.warpPerspective(
            src,self.M_in2out,(self.long_d,self.short_d),flags=cv2.INTER_LINEAR)
        
        possible_edges, sup_v = self.getPossibleEdge(dst)
        pt8 = []

        for i, pedge in enumerate(possible_edges):
            pedge = pedge.astype(np.float64)
            pedge -= np.min(pedge)
            pedge /= np.max(pedge)

            dysq = cv2.Sobel(pedge,-1,0,1,ksize=5)
            dysq = dysq**2

            std_thres = 0.1
            thres = std_thres*np.std(dysq) + np.mean(dysq)
            left_bias, right_bias = self.linear_regression2(dysq,thres)
            
            if left_bias != 0 or right_bias != 0:
                if i== 0:
                    pt1 = [sup_v[0],left_bias]
                    pt2 = [sup_v[1]-1,right_bias]
                elif i==1:
                    pt1 = [sup_v[0],sup_v[3]+left_bias]
                    pt2 = [sup_v[1]-1,sup_v[3]+right_bias]
                elif i==2:
                    pt1 = [left_bias,sup_v[3]-1]
                    pt2 = [right_bias,sup_v[2]]
                else:
                    pt1 = [sup_v[1]+left_bias,sup_v[3]-1]
                    pt2 = [sup_v[1]+right_bias,sup_v[2]]
            else:
                pt1 = [-1,-1]
                pt2 = [-1,-1]
            pt8.append(pt1)
            pt8.append(pt2)
        
        pt8 = np.array(pt8).reshape(4,4)
        
        A = np.zeros((8,2))
        C = np.zeros((8,1))
        corners = np.ones((3,4))
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
                
            corners[:2,0] = np.matmul(np.linalg.inv(A[0:2]),C[0:2]).reshape(2)
            corners[:2,1] = np.matmul(np.linalg.inv(A[6:8]),C[6:8]).reshape(2)
            corners[:2,2] = np.matmul(np.linalg.inv(A[2:4]),C[2:4]).reshape(2)
            corners[:2,3] = np.matmul(np.linalg.inv(A[4:6]),C[4:6]).reshape(2)

            corners = np.matmul(self.M_out2in,corners)
            corners /= corners[2]
            corners = np.transpose(corners[:2,:].astype(np.int32))


        return points_found, corners


    