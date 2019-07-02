import cv2
import numpy as np

source = cv2.imread('../pic/1.jpg')
gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30,
                           minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(source, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # cv2.circle(source, (i[1], i[1]), 2, (0, 0, 255), 3)

cv2.imwrite('1.jpg', source)
cv2.imshow('hough_circles', source)
cv2.waitKey()
