import cv2
import numpy as np

### READ VIDEO
cap = cv2.VideoCapture("./train-selected/2.mp4")

### ITERATE THROUGH EACH FRAME
while True:
    ret, frame = cap.read()
    if not ret:
        break

    ### CONVERT TO HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ### DEFINE LOWER AND UPPER BOUNDARIES OF THE METAL GRAY INSTRUMENTS
    lower_gray = np.array([0, 0, 150])
    upper_gray = np.array([140, 160, 255])

    ### CREATE MASK FOR METAL GRAY INSTRUMENTS
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    ### REMOVE NOISE FROM MASK
    kernel = np.ones((5, 5), np.uint8)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel) # erosion followed by dilation, removing the pixels that are too small and then dilating the actual objects
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel) # dilation followed by erosion, it will fill in the holes within the identified object and then clean up the borders

    ### FIND CONTOURS OF METAL GRAY INSTRUMENTS
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ### DRAW BOX AROUND METAL INSTRUMENTS
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:  # Adjust this threshold as needed
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 3)

    ### CREATE BLACK BACKGROUND WITH INSTRUMENTS
    foreground = cv2.bitwise_and(frame, frame, mask=mask_gray)

    ### STACK ONLY THE BOTTOM ROW OF IMAGES SIDE BY SIDE
    bottom_row_frame = frame[frame.shape[0]//2:, :]
    bottom_row_foreground = foreground[frame.shape[0]//2:, :]
    stacked = np.hstack((bottom_row_frame, bottom_row_foreground))

    ### SHOW IMAGE
    cv2.imshow("stacked", cv2.resize(stacked, None, fx=0.4, fy=0.4))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()