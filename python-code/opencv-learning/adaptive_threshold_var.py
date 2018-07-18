import numpy as np
import cv2
from BoxFilter import boxFilter, boxFilter_MeanStd


def adaptive_threshold_var(src, maxValue=255,  # 二值化，线性表中只有2个值 要么0 要么就是maxValue
                       blockSize=7, absThreshold=3, stdDevScale=0.0, debug=True):
    assert (blockSize % 2 == 1 and blockSize > 1)
    height, width = src.shape
    if debug:
        print(width, height)

    dst = src.copy()

    if maxValue < 0:
        # 二值化后值小于0，图像都为0
        dst[...] = 0
        return dst

    # 计算平均值和标准差作为比较值
    mean, stdv = boxFilter_MeanStd(src, blockSize)

    imaxval = np.clip(maxValue, 0, 255)

    for i in range(height):
        for j in range(width):
            idelta = max(stdDevScale * stdv[i, j], absThreshold)
            if int(src[i, j]) - int(mean[i, j]) > -idelta:
                dst[i, j] = imaxval
            else:
                dst[i, j] = 0
    return dst


if __name__ == "__main__":
    src = cv2.imread('datas/f3.jpg', 0)
    src = cv2.resize(src, (400, 300), interpolation=cv2.INTER_AREA)
    import time
    st = time.time()
    bw1 = adaptive_threshold_var(src, blockSize=201, absThreshold=10, stdDevScale=1.4, debug=False)
    st1 = time.time()
    print(time.time() - st1)

    cv2.imshow("1", bw1)
    cv2.waitKey(0)