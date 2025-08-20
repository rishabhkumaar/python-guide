import cv2
import numpy as np
from collections import deque

# Webcam
cap = cv2.VideoCapture(0)

# Canvas
ret, frame = cap.read()
canvas = np.zeros_like(frame)

# Narrowed Orange HSV Range (to avoid red shirt confusion)
lower_orange = np.array([8, 150, 150])
upper_orange = np.array([20, 255, 255])

# State
draw_mode = "pen"  # "pen", "eraser"
pen_color = (0, 0, 255)  # Red ink
pen_thickness = 6

# For smoothing cursor
pts = deque(maxlen=8)  # last few points

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for orange object
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cursor = None
    if contours:
        # Biggest contour
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:  # ignore tiny dots
            (x, y, w, h) = cv2.boundingRect(c)
            cursor = (x + w // 2, y + h // 2)
            pts.append(cursor)

            # Draw cursor point
            cv2.circle(frame, cursor, 10, (255, 255, 255), -1)
            cv2.circle(frame, cursor, 5, (0, 165, 255), -1)  # Orange center

            # Smooth cursor movement
            if len(pts) > 1 and pts[-2] is not None:
                if draw_mode == "pen":
                    cv2.line(canvas, pts[-2], pts[-1], pen_color, pen_thickness)
                elif draw_mode == "eraser":
                    cv2.line(canvas, pts[-2], pts[-1], (0, 0, 0), 40)

    else:
        pts.clear()  # reset if object lost

    # Tool Buttons (Pen, Erase, Clear)
    cv2.rectangle(frame, (10, 10), (110, 60), (0, 0, 255), -1)  # Pen
    cv2.putText(frame, "PEN", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(frame, (130, 10), (230, 60), (255, 255, 255), -1)  # Eraser
    cv2.putText(frame, "ERASE", (135, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.rectangle(frame, (250, 10), (350, 60), (0, 255, 0), -1)  # Clear
    cv2.putText(frame, "CLEAR", (260, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Check cursor inside buttons
    if cursor:
        x, y = cursor
        if 10 < x < 110 and 10 < y < 60:
            draw_mode = "pen"
        elif 130 < x < 230 and 10 < y < 60:
            draw_mode = "eraser"
        elif 250 < x < 350 and 10 < y < 60:
            canvas = np.zeros_like(frame)

    # Overlay drawing
    combined = cv2.add(frame, canvas)

    cv2.imshow("Digital Pen", combined)
    cv2.imshow("Mask", mask)  # Debugging mask

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
