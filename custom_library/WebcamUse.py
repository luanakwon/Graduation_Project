import cv2
cap = cv2.VideoCapture(0)

print(cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while cv2.waitKey(33) < 0:
    ret, frame = cap.read()
    frame = cv2.flip(frame,flipCode=1)
    cv2.imshow("VideoFrame", frame)
cv2.imwrite('kakaocard.png',frame)
cap.release()
cv2.destroyAllWindows()