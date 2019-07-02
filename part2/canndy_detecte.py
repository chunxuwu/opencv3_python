import cv2

# canny 边缘检测算法分为5个步骤，用高斯滤波器对图像进行去噪，计算梯度，在边缘使用非最大抑制，
# 在检测边缘上使用双阈值去除假阳性，最后还会计算所有的边缘与其之间的连接
img = cv2.imread('../pic/1.jpg', 0)
cv2.imwrite('canny.jpg', cv2.Canny(img, 200, 300))
cv2.imshow('canny',cv2.imread('canny.jpg'))
cv2.waitKey()
cv2.destroyWindow()
