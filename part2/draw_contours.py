import cv2
import numpy as np

# 下采样读取图片，清晰度默认为原来的1/4
img = cv2.pyrDown(cv2.imread('../pic/1.jpg', cv2.IMREAD_UNCHANGED))
# 将图片转为灰度图片，然后将像素大于127的置为255。其他的置为0，变为二值图片。返回的ret为阈值，thresh为阈值化后的图片
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),
                            127, 255, cv2.THRESH_BINARY)

contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # 在img上画矩形框
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # 省略小数点后的数字，不是四舍五入
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 1)

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # cast to image
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    # img = cv2.circle(img, center, radius, (0, 255, 0), 1)
# img 为图片，contours为轮廓，-1表示绘制所有轮廓（否则为指定轮廓），然后指定颜色和线宽
# cv2.drawContours(img, contours, 100, (255, 0, 0), 1)
cnt = contours[100]
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.drawContours(img, [approx], -1, (255, 0, 0), 1)
# 用点描绘轮廓
# cv2.drawContours(img, approx, -1, (255, 0, 0), 1)
# 轮廓形状
hull = cv2.convexHull(cnt)
cv2.drawContours(img, [hull], -1, (0, 255, 0), 1)

cv2.imshow('contours', img)
cv2.waitKey()

