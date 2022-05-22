import time

import cv2
import mouse
import numpy as np
from screeninfo import get_monitors

import HandTrackingModule as Htm

cam_width, cam_height = 1280, 960
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height
active_region = [(200, 100), (cam_width - 200, cam_height - 550)]  # Active hand tracking region
region_offset = 50  # Offset in pixels
active_region_offset = [(active_region[0][0] - region_offset, active_region[0][1] - region_offset),
                        (active_region[1][0] + region_offset, active_region[1][1] + region_offset)]

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = Htm.HandDetector(max_hands=1)

previous_time = 0

invert = True

click_latch = False
drag_latch = False

damping = 4
previous_x, previous_y = 0, 0
current_x, current_y = 0, 0

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.find_hands(img, draw=True)
    lm_list, bounding_box = detector.find_position(img, draw_lm=False)

    # 2. Get the tip of the thumb and index finger
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]

        # 3. Check which fingers are up
        fingers = detector.fingers_up()

        # 4. If index and thumb are out, MOVING MODE
        if active_region_offset[0][0] < x1 < active_region_offset[1][0] and \
                active_region_offset[0][1] < y1 < active_region_offset[1][1]:

            if fingers[0] == 1 and fingers[1] == 1:
                # 5. Convert coordinates
                cv2.rectangle(img, active_region[0], active_region[1], (255, 0, 255), 2)
                cv2.rectangle(img, active_region_offset[0], active_region_offset[1], (255, 0, 0), 2)
                cx = np.interp(x1, (active_region[0][0], active_region[1][0]),
                               ((screen_width, 0) if invert else (0, screen_width)))
                cy = np.interp(y1, (active_region[0][1], active_region[1][1]), (0, screen_height))

                # 6. Smoothen values
                current_x = previous_x + (cx - previous_x) / damping
                current_y = previous_y + (cy - previous_y) / damping

                if fingers[1] == 1 and fingers[4] == 1 and not drag_latch:
                    # 7. If pinkie goes up, DRAG
                    drag_latch = True
                    mouse.press()
                elif fingers[1] == 1 and fingers[4] == 0 and drag_latch:
                    mouse.release()
                    drag_latch = False
                else:
                    # 8. Move mouse
                    mouse.move(current_x, current_y)

                previous_x, previous_y = current_x, current_y

            # 9. If thumb goes in, CLICK
            if fingers[0] == 0 and fingers[1] == 1 and not click_latch:
                if fingers[2] == 1 and fingers[3] == 1:
                    mouse.right_click()
                elif fingers[2] == 1:
                    mouse.double_click()
                else:
                    mouse.click()
                click_latch = True
            elif fingers[0] == 1 and fingers[1] == 1 and click_latch:
                click_latch = False

    # 10. Frame rate
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 11. Display
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
