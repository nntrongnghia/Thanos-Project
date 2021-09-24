import cv2
import os
from dataset_config import IPN_HAND_ROOT

video_dir = os.path.join(IPN_HAND_ROOT, "videos")
cap = cv2.VideoCapture(os.path.join(video_dir, "1CM1_1_R_#218.avi"))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("video", frame)
        if cv2.waitKey(20) == 27: # ESC key
            break
cv2.destroyAllWindows()
cap.release()