import cv2
import numpy as np

img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 255
# ret为阈值，thresh为返回的图像，
ret, thresh = cv2.threshold(img, 127, 255, 1)
#
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0, 250, 2), 2)
cv2.imshow('contour', color)
cv2.waitKey()
cv2.destroyWindow()