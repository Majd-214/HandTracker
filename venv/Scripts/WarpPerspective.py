import time
import cv2
import mouse
import numpy as np
from screeninfo import get_monitors
import HandTrackingModule as Htm

cam_width, cam_height = 1280, 960
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

image_width, image_height = 250, 350

img = cv2.imread("playing_cards.jpg")

# src = np.float32([[729, 193], [975, 273], [551, 421], [814, 523]])
src = np.float32([[814, 523], [551, 421], [975, 273], [729, 193]])
dst = np.float32([[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]])

matrix = cv2.getPerspectiveTransform(src, dst)
img_out = cv2.warpPerspective(img, matrix, (image_width, image_height))

for x in range(0, 4):
    cv2.circle(img, (int(src[x][0]), int(src[x][1])), 5, (0, 255, 0), cv2.FILLED)

cv2.imshow("Original Image", img)
cv2.imshow("Transformed Image", img_out)
cv2.waitKey(0)
