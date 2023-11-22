import numpy as np
import pywt
import cv2


def w2d(img, mode="haar", level=1):
    imArray = img
    # convert img to gray scale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # computing coefficients and removing low frequency components
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeff_H = list(coeffs)
    coeff_H[0] *= 0
    ## reconstruction
    imArray_H = pywt.waverec2(coeff_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H
