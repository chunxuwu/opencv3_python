import cv2
clicked = False


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True


camera = cv2.VideoCapture(0)
cv2.namedWindow('my_window')
cv2.setMouseCallback('my_window', onMouse)
print('click window or press any key to stop')
# waitKey(1)==-1表示触发时间为1毫秒，没有按键被按
while cv2.waitKey(1) == -1 and not clicked:
    success, frame = camera.read()
    cv2.imshow('my_window', frame)
cv2.destroyAllWindows()
camera.release()
