import cv2
import numpy
import time


class CaptureManager(object):
    def __init__(self, capture, preview_windows_manager=None,
                 should_mirror_preview=False):

        self.previewWindowsManager = preview_windows_manager
        self.shouldMirrorPreview = should_mirror_preview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._frameElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
            return self._frame

    @property
    def is_writing_image(self):
        return self._imageFilename is not None

    @property
    def  is_writing_video(self):
        return self._videoFilename is not None

    def enter_frame(self):
        # 判断 enterFrame为True时，报后面的错，为false时，直接过
        assert not self._enteredFrame, 'previous enterFrame() had no Matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exit_frame(self):
        if self.frame is None:
            self._enteredFrame = False

        # update fps and related variables
        if self._frameElapsed == 0:
            self._startTime = time.time()
        else:
            time_elapsed = time.time()-self._startTime
            self._fpsEstimate = self._frameElapsed/time_elapsed
        self._frameElapsed += 1
        # draw to the windows
        if self.previewWindowsManager is not None:
            if self.shouldMirrorPreview:
                mirrored_frame = numpy.fliplr(self._frame).copy()
                self.previewWindowsManager.show(mirrored_frame)
            else:
                self.previewWindowsManager.show(self._frame)
        # write to image file, if any
        if self.is_writing_image:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None
        #
        self._writeVideoFrame()

        self._frame = None
        self._enteredFrame = False

    def _writeVideoFrame(self):
        if not self.is_writing_video:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._frameElapsed < 20:
                    return
                else:
                    fps = self._fpsEstimate

            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)
        self._videoWriter.write(self._frame)

    def writeImage(self, filename):
        """write the next exited frame to an image file"""
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """start writing exited frames to a video file"""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """stop writing exited frames to a video file"""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None


# 创建窗口管理函数
class WindowManager(object):
    def __init__(self, windowsName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowsName
        self._isWindowCreated = False

    @property
    def isWindowsCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    # close windows
    def destroyWindows(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        key_code = cv2.waitKey(1)
        if self.keypressCallback is not None and key_code != -1:
            key_code &= 0xFF
            self.keypressCallback(key_code)
