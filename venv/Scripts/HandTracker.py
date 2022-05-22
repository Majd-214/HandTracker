import time
import cv2
import HandTrackingModule as Htm

cap = cv2.VideoCapture(0)
detector = Htm.HandDetector(max_hands=1, detection_con=0.6)
previous_time = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img, True)
    pos_list = detector.find_position(img, draw_lm=False)

    if len(pos_list) != 0:
        print(pos_list[8])

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
