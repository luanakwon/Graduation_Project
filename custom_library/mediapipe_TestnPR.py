import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('mytestvideo.mp4')

resolution = (450,800)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outVideo.mp4',fourcc,30.0,resolution)


with mp_hands.Hands(
    model_complexity= 0,
    min_detection_confidence=0.5,
    min_tracking_confidence= 0.5) as hands:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
  

        img = cv2.resize(img, resolution)
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
        
        img = cv2.flip(img, 1)
    
        
        out.write(img)
