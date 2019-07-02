import numpy as np
import cv2
from matplotlib import pyplot as plt

# 使用分水岭算法进行图像分割
img = cv2.imread('../pic/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 将颜色转为灰度后，可为图像设一个阈值，将图像分为两部分：黑色部分和白色部分
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal 噪声去除，morphologyEx是一种对图像进行膨胀之后再进行腐蚀的操作
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area 确定背景区域，图像进行膨胀操作
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area，通过distanceTransform来获取确定的前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

plt.imshow(img)
plt.show()
