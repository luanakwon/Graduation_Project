import numpy as np
import mediapipe as mp

class FingerDipDetector():
    def __init__(self,
        static_image_mode=True,
        max_num_hands=1,
        model_complexity= 0,
        min_detection_confidence=0.5):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence)

    def run(self,img):
        finger_dips = np.zeros((4,2),dtype=np.int32)
        finger_dirs = np.zeros((4,2),dtype=np.float64)
        ret = False
        height, width = img.shape[:2]

        results = self.hands.process(img)

        if results.multi_hand_landmarks:
            ret = True
            for hlm in results.multi_hand_landmarks:
                for i, p_idx in enumerate([6,10,14,18]):
                    finger_dips[i] = np.int32(
                        [hlm.landmark[p_idx].x*width,
                        hlm.landmark[p_idx].y*height]
                    )
                    finger_dirs[i] = np.float64(
                        [(hlm.landmark[p_idx+1].x-hlm.landmark[p_idx].x)*width,
                        (hlm.landmark[p_idx+1].y-hlm.landmark[p_idx].y)*height]
                    )
                break
            

        return ret,finger_dips,finger_dirs

