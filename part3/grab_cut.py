import numpy as np
import cv2
from matplotlib import pyplot as plt

# 使用分水岭和GrabCut算法进行物体分割
img = cv2.imread('../pic/1.jpg')

# img.shape=(1039, 690, 3)
# img.shape[0:2]=(1039, 690)
mask = np.zeros(img.shape[:2], np.uint8)
a = img.shape[:2]
# 背景色bgdModel，前景色fgdModel
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 感兴趣区域ROI的x，y，宽度，高度
rect = (1, 1, 700, 700)

# 获得返回值mask、bgdModel、fgdModel。
# 目标图像、掩码、感兴趣区域，背景、前景、算法迭代次数、操作模式
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 经过图像分割法grabCut处理之后，
# print(set(mask.ravel())) -> {0,2,3}
# mask的掩码元素{0}->{0,2,3}
# where(condition,x,y)，condition为array_like或bool
# 真yield x，假yield y
# mask等于0和2的地方置0，否则置1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')


# 两者乘积则报错：
# 操作数无法与形状一起广播
# ValueError: operands could not be broadcast together with shapes (1039,690,3) (1039,690)
# 为了保持数形一致，增加np.newaxis
# mask2[:,:,np.newaxis].shape=(1039,690,1)
# 这样当行列值不相等时可进行广播计算
# 经过计算后，将背景色赋值为0，即为黑色
img2 = img * mask2[:, :, np.newaxis]

# subplot(121)创建1行2列，当前位置为1
plt.subplot(121), plt.imshow(img2)
plt.title("grabcut")
# subplot(122)当前位置为2
plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])
plt.show()
