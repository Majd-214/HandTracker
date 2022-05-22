import cv2
import time
import os
import HandTrackingModule as Htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folder_path = "FingerImages"
my_list = os.listdir(folder_path)
print(my_list)
overlay_list = []
for im_path in my_list:
    image = cv2.imread(f'{folder_path}/{im_path}')
    # print(f'{folder_path}/{im_path}')
    overlay_list.append(image)

print(len(overlay_list))
previous_time = 0

detector = Htm.HandDetector(detection_con=0.75)

fingertips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmarks, bounding_box = detector.find_position(img, draw_lm=False)
    # print(landmarks)

    if len(landmarks) != 0:

        # print(fingers)
        total_fingers = detector.fingers_up().count(1)
        print(total_fingers)

        h, w, c = overlay_list[total_fingers - 1].shape
        img[0:h, 0:w] = overlay_list[total_fingers - 1]

        cv2.rectangle(img, (20, 275), (170, 475), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 425), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
