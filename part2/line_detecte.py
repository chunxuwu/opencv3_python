import cv2
import numpy as np
# 进行直线检测
img = cv2.imread('../pic/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 边缘检测
edges = cv2.Canny(gray, 50, 120)
min_line_length = 20
max_line_gap = 5
# edges为单通道二值图像，1和np.pi/180是图片的几何表示，然后是最小线长以及允许的线段之间的距离。
# 若线段间的距离大于5，则认为是俩条线段
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, min_line_length, max_line_gap)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('edges', edges)
cv2.imshow('lines', img)
cv2.waitKey()