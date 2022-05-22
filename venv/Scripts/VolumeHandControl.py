import time
import cv2
import math
import numpy as np
import HandTrackingModule as Htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol = volume.GetVolumeRange()[0]
max_vol = volume.GetVolumeRange()[1]

vol = 0
vol_bar = 400
vol_per = 0

cap = cv2.VideoCapture(0)
detector = Htm.HandDetector(max_hands=1, detection_con=0.6)
previous_time = 0

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.find_hands(img, True)
    pos_list = detector.find_position(img, draw_lm=False)
    if len(pos_list) != 0:

        # Filter based on size

        # Find distance between index and thumb

        # Convert Volume
        # Reduce resolution to increase usability
        # Check fingers Up/Down
        # If pinkie is down, set volume
        # Drawings
        # Frame rate

        # print(pos_list[4], pos_list[8])

        x1, y1 = pos_list[4][1], pos_list[4][2]
        x2, y2 = pos_list[8][1], pos_list[8][2]
        x2, y2 = pos_list[8][1], pos_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot((x2-x1), (y2-y1))
        vol = np.interp(length, [30, 200], [min_vol+10, max_vol])
        vol_bar = np.interp(length, [30, 200], [400, 150])
        vol_per = np.interp(length, [30, 200], [0, 100])

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

        print(length)

        if length < 30:  # Range [30 <---> 200]
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            vol = min_vol
        else:
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        volume.SetMasterVolumeLevel(vol, None)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)
    cv2.putText(img, f'{int(vol_per)}%', (40, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
