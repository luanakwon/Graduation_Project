import cv2
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

resolution = (480,640)

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity= 0,
    min_detection_confidence=0.5,) as hands:
    
    while cv2.waitKey(33) < 0:
        ret, frame = cap.read()
        if not ret:
            break

        # mirror the frame
        img = cv2.flip(frame,flipCode=1)

        # change to portrait of ratio 800:600
        img = cv2.resize(img[:,140:500],dsize=resolution)

        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for handedness in results.multi_handedness:
                print(handedness)
            
        #         for p_idx in [6,10,14,18]:
        #             print(hand_landmarks.landmark[p_idx])
        #             anc_x = hand_landmarks.landmark[p_idx].x
        #             anc_y = hand_landmarks.landmark[p_idx].y
        #             anc_x = int(anc_x*resolution[0]-ROI_win_size/2)
        #             anc_y = int(anc_y*resolution[1]-ROI_win_size/2)
        #             ROI_mask[anc_y:anc_y+ROI_win_size,anc_x:anc_x+ROI_win_size] = 1
        
        # masked_img = img*ROI_mask

        #img = cv2.flip(img, 1)
        cv2.resize(img,dsize=(0,0),fx=0.5,fy=0.5)
        cv2.imshow('title',img)

    
    