import numpy as np
from cv2 import cv2

import sys
import matplotlib.pyplot as plt

# Gaussian filter
def gaussian_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    # Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    # prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out


def  sharpen(image):
    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
    dst = cv2.filter2D(image, -1, kernel=kernel)
    # cv2.imshow("result", dst)
    # cv2.waitKey(0)
    return dst


def enhancement(image):
    # 将图像分成三通道,对每个图像均值化
    (R, G, B) = cv2.split(image)
    r = cv2.equalizeHist(R)  # 灰度图像直方图均衡化
    b = cv2.equalizeHist(B)  # 灰度图像直方图均衡化
    g = cv2.equalizeHist(G)  # 灰度图像直方图均衡化
    merged = cv2.merge([r, g, b])
    # cv2.imshow("Merged", merged)
    # cv2.waitKey()
    return merged


def normalize(merged):
    out1 = np.zeros(merged.shape, np.uint8)
    cv2.normalize(merged, out1, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imshow("out", out1)
    # cv2.waitKey()
    return out1


def pre_process(image):
    a = gaussian_filter(image, K_size=3, sigma=1.3)
    b = sharpen(a)
    c = enhancement(b)
    d = normalize(c)
    return d

if __name__ == '__main__':
    
    image = cv2.imread('sample.jpg')
    image = pre_process(image)
    cv2.imwrite("testbak.jpg", image)
    # cv2.imshow("final",x)
    # cv2.waitKey(0)