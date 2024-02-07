import cv2
import numpy as np 

cap = cv2.VideoCapture("./cars_video.mp4")

### CREATE BACKGROUND OBJECT
backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2) #remove the background, grayscale, MOG2 can detect shadows+objects
kernel = np.ones((3,3), np.int8) #kernels are how much the value of a pixel should change based on the values of surrounding pixels
kernel2 = None

### FOR EACH FRAME
while True:
    #iterates through each frame of the video
    ret, frame = cap.read()
    if not ret:
        break

    ### FIND LOCATION OF OBJECTS
    fgmask = backgroundObject.apply(frame) #apply ln 7 to selected frame
    _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY) #turns img to black+white and better separates background+objects, values < 20 = 0 (black) and values > 20 = 255 (white)

    ## using both erotion and dilation can remove small objects and smooth the border for large objects
    fgmask = cv2.erode(fgmask, kernel, iterations=1) #identifes the border of objects and then removes the excess pixels around it
    fgmask = cv2.dilate(fgmask, kernel2, iterations=6) #adds pixels to the boundaries of objects in an image

    ### CREATE THE BOXES
    countors, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find countours draws lines around borders. chain_approx_simple saves memory by only storing the endpoints of boundaries instead of storing each point on the border

    frameCopy = frame.copy()

    for cnt in countors:
        if cv2.contourArea(cnt) > 5000: # if the object is big enough
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x,y), (x+width, y+height), (0,0,255), 2)
            cv2.putText(frameCopy, "Car detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

    ### COMBINE EVERYTHING TOGETHER
    foreground = cv2.bitwise_and(frame, frame, mask=fgmask)
    stacked = np.hstack((frameCopy, foreground))
    
    ### SHOW IMAGE
    cv2.imshow("stacked", cv2.resize(stacked, None, fx=0.4, fy=0.4))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()