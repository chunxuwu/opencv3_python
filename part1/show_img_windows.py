import cv2
import numpy as np

img = cv2.imread('../pic/1.jpg')
cv2.imshow('my image', img)
# 显示1秒
cv2.waitKey(1000)
# 视频由opencv创建的窗口
cv2.destroyAllWindows()
