import cv2


# 图片识别方法封装,
def discern(img):
    # 将图img转为灰度图，因opencv中的人脸检测是基于灰度的色彩空间
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 实例化级联分类器
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    # 调用图像中的对象，并返回矢量矩形
    faceRects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    # 只要视频存在，就输出人脸识别框
    if len(faceRects):
        # 框出人脸，分开写
        # for faceRect in faceRects:
        # x, y, w, h = faceRect
        # cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
        # 框出人脸，合并写
        for (x, y, w, h) in faceRects:
            # 在原彩色图img上绘制识别的人脸矩形
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
    # 将绘制有矩形框的img显示输出
    cv2.imshow("Image", img)


# 打开一个VideoCapture目标（初始化摄像头）
# 参数为摄像头ID，0表获取第一个摄像头
cap = cv2.VideoCapture(0)
while (True):
    # 将视频逐帧显示
    ret, img = cap.read()
    # 调用discern函数
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头
cap.release()
# 释放窗口资源
cv2.destroyAllWindows()
