import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('mytestvideo2.mp4')

resolution = (1080,1920)
ROI_win_size = 270
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('outVideo2.mp4',fourcc,30.0,resolution)

ROI_mask = np.zeros((1920,1080,3),dtype=np.uint8)

with mp_hands.Hands(
    model_complexity= 0,
    min_detection_confidence=0.5,
    min_tracking_confidence= 0.5) as hands:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
  

        img = cv2.resize(img, resolution)
        ROI_mask = np.zeros_like(img)

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
            
                for p_idx in [6,10,14,18]:
                    print(hand_landmarks.landmark[p_idx])
                    anc_x = hand_landmarks.landmark[p_idx].x
                    anc_y = hand_landmarks.landmark[p_idx].y
                    anc_x = int(anc_x*resolution[0]-ROI_win_size/2)
                    anc_y = int(anc_y*resolution[1]-ROI_win_size/2)
                    ROI_mask[anc_y:anc_y+ROI_win_size,anc_x:anc_x+ROI_win_size] = 1
        
        img *= ROI_mask

        #img = cv2.flip(img, 1)
        cv2.resize(img,dsize=(0,0),fx=0.5,fy=0.5)
        cv2.imshow('title',img)
        cv2.waitKey()
    
    