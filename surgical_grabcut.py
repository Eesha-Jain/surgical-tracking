import cv2
import numpy as np

### READ VIDEO
cap = cv2.VideoCapture("./train-selected/c7v1.mp4")

### ITERATE THROUGH EACH FRAME
while True:
    ret, frame = cap.read()
    if not ret:
        break

    ### RESIZE FRAME
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    ### APPLY GRABCUT TO REFINE THE MASK
    mask_grabcut = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (5, 5, frame.shape[1]-5, frame.shape[0]-5)  # Adjusted rectangle size
    cv2.grabCut(frame, mask_grabcut, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)  # Reduced number of iterations

    ### MODIFY THE MASK FROM GRABCUT
    mask_grabcut = np.where((mask_grabcut==2)|(mask_grabcut==0), 0, 1).astype('uint8') # Adjusted mask values

    ### Convert ROI obtained from GrabCut to HSV
    hsv_roi = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask_grabcut), cv2.COLOR_BGR2HSV)

    ### DEFINE LOWER AND UPPER BOUNDARIES OF THE METAL GRAY INSTRUMENTS IN HSV
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([179, 50, 255])

    ### APPLY HSV FILTERING TO THE GRABCUT MASK
    mask_gray = cv2.inRange(hsv_roi, lower_gray, upper_gray)

    ### REMOVE NOISE FROM MASK
    kernel = np.ones((3, 3), np.uint8)  # Optimized kernel size
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)

    ### FIND CONTOURS OF METAL GRAY INSTRUMENTS
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ### DRAW BOX AROUND METAL INSTRUMENTS
    frameCopy = frame.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            #cv2.drawContours(frameCopy, [cnt], -1, (0,255,0), 3)
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x,y), (x+width, y+height), (0,0,255), 2)

    ### CREATE BLACK BACKGROUND WITH INSTRUMENTS
    foreground = cv2.bitwise_and(frame, frame, mask=mask_gray)

    ### STACK ONLY THE BOTTOM ROW OF IMAGES SIDE BY SIDE
    bottom_row_frame = frameCopy[frameCopy.shape[0]//2:, :]
    bottom_row_foreground = foreground[frame.shape[0]//2:, :]
    stacked = np.hstack((bottom_row_frame, bottom_row_foreground))

    ### SHOW IMAGE
    cv2.imshow("stacked", cv2.resize(stacked, None, fx=0.4, fy=0.4))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()