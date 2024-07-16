# because the background is moving and hence has a velocity, the kalman filter also adds a contour around the pink background after the hsv does its filtering, which is not ideal

import cv2
import numpy as np

### READ VIDEO
cap = cv2.VideoCapture("./train-selected/c7v1.mp4")

### KALMAN FILTER INITIALIZATION
kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32) #indicates that you want the x,y position of the object
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32) #says dx influences x, dy influences y, dx & dy r not influenced by anything

### INITIALIZE VARIABLES FOR KALMAN FILTER
last_measurement = np.array((2, 1), np.float32)  # Last measurement (x, y)
last_prediction = np.zeros((2, 1), np.float32)  # Last prediction (x, y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([179, 50, 255])

    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    kernel = np.ones((5, 5), np.uint8)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            ### KALMAN FILTER PREDICTION
            prediction = kalman.predict()
            x, y, w, h = cv2.boundingRect(cnt)

            ### KALMAN FILTER CORRECTION
            measurement = np.array([[x + w / 2], [y + h / 2]], np.float32)
            if measurement[0][0] != 0 and measurement[1][0] != 0:
                kalman.correct(measurement)
                last_measurement = measurement
            else:
                measurement = last_measurement

            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)

    foreground = cv2.bitwise_and(frame, frame, mask=mask_gray)

    bottom_row_frame = frame[frame.shape[0]//2:, :]
    bottom_row_foreground = foreground[frame.shape[0]//2:, :]
    stacked = np.hstack((bottom_row_frame, bottom_row_foreground))

    cv2.imshow("stacked", cv2.resize(stacked, None, fx=0.4, fy=0.4))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()