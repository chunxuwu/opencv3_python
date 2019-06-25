import numpy
import cv2
import os
# img = numpy.zeros((3, 3), dtype=numpy.uint8)
# print(img)
# 将灰度图像转化为BGR
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# print(img.shape)

# 读取jpg图片，保存为png
img = cv2.imread('./pic/1.jpg')
# cv2.imwrite('2.png', img)

# random_byte_array = bytearray(os.urandom(120000))
# numpy_array = numpy.array(random_byte_array)

# grayImage = numpy_array.reshape(300, 400)
# cv2.imwrite('./pic/random_gray.png', grayImage)
#
# bgrImage = numpy_array.reshape(100, 400, 3)
# cv2.imwrite('./pic/random_bgr.png', bgrImage)

# num = numpy.random.randint(0, 256, 120000)
# bgr_img = num.reshape(100, 400, 3)
# cv2.imwrite('./pic/bgr_img.jpg', bgr_img)

# 使用numpy.array访问图像数据
# 把左上角的像素变白
img[0, 0] = [255, 255, 255]
print(img.item(150, 120, 1))
# 设置某一位置的像素
img.itemset((150, 120, 1), 255)
print(img.item(150, 120, 1))
# 将三元数组设为0，即Blue为0
img[:, :, 2] = 0
# 复制图片的某些区域
my_roi = img[0:100, 0:100]
img[300:400, 300:400] = my_roi

# 视频文件的读写
video = cv2.VideoCapture('./video/1.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 采用未压缩的yuv颜色编码，是4：2：0色度子采样，兼容性好，但产生的文件大
# video_writer = cv2.VideoWriter('2.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
# success, frame = video.read()
# 编码类型为MPEG-1
# video_writer = cv2.VideoWriter('1.avi', cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), fps, size)
# 编码类型为MPEG-4
# video_writer = cv2.VideoWriter('3.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
video_writer = cv2.VideoWriter('4.flv', cv2.VideoWriter_fourcc('F', 'L', 'V', '1'), fps, size)
success = True
while success:
    success, frame = video.read()
    video_writer.write(frame)



