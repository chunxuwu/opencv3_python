import cv2
camera_capture = cv2.VideoCapture(0)
fps = 30
size = (int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writer = cv2.VideoWriter('1.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
num_frame = 10*fps-1
while num_frame > 0:
    success, frame = camera_capture.read()
    video_writer.write(frame)
    num_frame -= 1
camera_capture.release()
