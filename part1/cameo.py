import cv2
from managers import WindowManager, CaptureManager
from part2 import filters


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = filters.SharpenFilter()

    def run(self):
        """run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowsCreated:
            self._captureManager.enter_frame()
            frame = self._captureManager.frame
            filters.stokeEdges(frame, frame)
            self._curveFilter.apply(frame, frame)

            self._captureManager.exit_frame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        handle a keypress
        space  take a screen shot
        tab start/stop recording a screencast
        escape quit
        :param keycode:
        :return:
        """
        if keycode == 32:  # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.is_writing_video:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindows()


if __name__ == '__main__':
    Cameo().run()
