# coding:utf-8
import numpy as np
import cv2
from BoxFilter import boxFilter
import time

def adaptive_threshold(src, maxValue=255, blockSize=7, delta=3, debug=False):
    assert(blockSize % 2 == 1 and blockSize > 1)
    height, width = src.shape
    if debug:
        print(width, height)

    dst = src.copy()

    if maxValue < 0:
        dst[...] = 0
        return dst

    # 计算平均值作为比较值
    # mean = cv2.boxFilter(src, -1, (blockSize, blockSize), normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean = boxFilter(src, blockSize)

    imaxval = np.clip(maxValue, 0, 255)
    idelta = delta
    tab = np.zeros(768, dtype=src.dtype)
    # 构建查找表，index 就是像素值的大小（key），对应的值就是阈值比较过后应该是0还是255
    for i in range(768):
        if i - 255 > -idelta:
            tab[i] = imaxval
        else:
            tab[i] = 0

    if debug:
        print("tab:", tab, tab.shape)

    # dst = src.astype(int) - mean.astype(int) + 255
    # zz = cv2.LUT(dst, tab)
    # 逐像素计算src[j] - mean[j] + 255，并查表得到结果
    for i in range(height):
        for j in range(width):
            dst[i, j] = tab[int(src[i, j]) - int(mean[i, j]) + 255]
    return dst


if __name__ == "__main__":
    src = cv2.imread("datas/f3.jpg")
    src = cv2.resize(src, (400, 300), interpolation=cv2.INTER_AREA)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    st = time.time()
    bw = adaptive_threshold(src_gray, blockSize=201, delta=30)
    print("elapsed: ", time.time() - st)
    cv2.imshow("1", bw)
    cv2.waitKey(0)