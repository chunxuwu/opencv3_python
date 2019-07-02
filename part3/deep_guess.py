import numpy as np
import cv2


def update(val=0):
    # disparity range is tuned for 'aloe' image pair
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowsize', 'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    print('computing disparity...')

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv2.imshow('left', imgL)
    cv2.imshow('right', imgR)
    cv2.imshow('disparity2', (disp - min_disp) / num_disp)


if __name__ == '__main__':
    window_size = 5
    min_disp = 16
    num_disp = 192 - min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400
    # 加载两幅图
    imgL = cv2.imread('../pic/2.jpg')
    imgL = cv2.resize(imgL, (500, 500))
    imgR = cv2.imread('../pic/3.jpg')
    imgR = cv2.resize(imgR, (500, 500))
    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)

    # 创建一个StereoSGBM实例，是一种计算视图差的算法
    # 并创建几个跟踪条来调整算法参数，然后调用update函数
    # update函数将跟踪条的值传给StereoSGBM实例
    # StereoSGBM是semiglobal block matching 的缩写
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    update()
    cv2.waitKey()
