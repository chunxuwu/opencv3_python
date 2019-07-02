import cv2
import numpy
import utils


def stokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 7:
        blurred_src = cv2.medianBlur(src, blurKsize)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize=edgeKsize)
    normalized_inverse_alpha = (255-gray_src)/255
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel*normalized_inverse_alpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """ a filter that applies a convolution to V"""
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """apply the filter with a bgr or gray source"""
        cv2.filter2D(src, -1, self._kernel, dst)


# 锐化
class SharpenFilter(VConvolutionFilter):
    """a shape filter with a 1-pixel radius"""
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """a blur filter with a 2-pixl radius"""
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """an emboss filter with a 1-pixl radius"""
    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 3]])
        VConvolutionFilter.__init__(self, kernel)