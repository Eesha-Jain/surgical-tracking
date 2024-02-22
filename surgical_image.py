import cv2
import numpy as np

### READ IMAGE
image = cv2.imread("./surgical-tracking/surgical_tools_photo.png")

### CONVERT TO HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

### DEFINE LOWER AND UPPER BOUNDARIES OF THE METAL GRAY INSTRUMENTS
lower_gray = np.array([0, 0, 50])
upper_gray = np.array([179, 50, 255])

### CREATE MASK FOR METAL GRAY INSTRUMENTS
mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

### REMOVE NOISE FROM IMAGE
kernel = np.ones((5,5), np.uint8)
mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel) #erosion followed by dilation, removing the pixels that are too small and then dilating the actual objects
mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel) #dilation followed by erosion, it will fill in the holes within the identified object and then clean up the borders

### FIND CONTOURS OF METAL GRAY INSTRUMENTS
contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

### DRAW BOX AROUND METAL INSTRUMENTS
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

### CREATE BLACK BACKGROUND WITH INSTRUMENTS
foreground = cv2.bitwise_and(image, image, mask=mask_gray)
foreground_black_bg = np.zeros_like(foreground)
foreground_black_bg[mask_gray > 0] = foreground[mask_gray > 0]

### STACK IMAGES SIDE BY SIDE
stacked = np.hstack((image, foreground_black_bg))

### SHOW IMAGE
cv2.imshow("Identifying instrument in image", cv2.resize(stacked, None, fx=0.4, fy=0.4))

cv2.waitKey(0)
cv2.destroyAllWindows()