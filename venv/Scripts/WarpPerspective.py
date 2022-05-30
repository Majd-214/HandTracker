import time
import cv2
import mouse
import numpy as np
from screeninfo import get_monitors
import HandTrackingModule as Htm

cam_width, cam_height = 1024, 576
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

# image_width, image_height = 1920, 1080
# img = cv2.imread("Resources/demo.jpg")

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = Htm.HandDetector(max_hands=1, detection_con=0.2, track_con=0.3)

DIM = (1024, 576)
K = np.array([[797.0118923274562, 0.0, 525.6367078328287], [0.0, 800.2609207364642, 280.0906945970121], [0.0, 0.0, 1.0]])
D = np.array([[-0.0031319388008956553], [-0.6096160899954018], [1.981574230203118], [-2.523538885945794]])

src = np.float32([[30, 50], [977, 50], [178, 519], [878, 506]])
# src = np.float32([[0, 43], [1027, 45], [136, 584], [921, 576]])

dst = np.float32([[0, 0], [cam_width, 0], [0, cam_height], [cam_width, cam_height]])

matrix = cv2.getPerspectiveTransform(src, dst)
matrix_fisheye = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=0.0)

while True:
    success, img = cap.read()
    img = cv2.fisheye.undistortImage(img, K, D, None, Knew=matrix_fisheye)
    img = cv2.resize(img, (1024, 576))
    # img = cv2.warpPerspective(img, matrix, DIM)
    img = detector.find_hands(img)
    landmarks, bounding_box = detector.find_position(img, draw_lm=False)

    if len(landmarks) != 0:
        # landmark = (landmarks[8][1], landmarks[8][2])
        landmark = np.array([[landmarks[8][1], landmarks[8][2]]], dtype=np.float32)
        # landmark = cv2.warpPerspective(landmark, matrix, (2, 1))
        p = (landmarks[8][1], landmarks[8][2])
        px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
        py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
        x = np.interp(int(px), (0, cam_width), (0, screen_width))
        y = np.interp(int(py), (0, cam_height), (0, screen_height))
        p_after = (int(x), int(y))
        mouse.move(int(x), int(y))
        print(p_after)
    for x in range(0, 4):
        cv2.circle(img, (int(src[x][0]), int(src[x][1])), 1, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Original Image", img)
    cv2.waitKey(1)
