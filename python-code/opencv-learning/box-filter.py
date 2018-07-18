# coding=utf-8
"""
Grayscale box filter.
USAGE: BoxFilter.py [--filter_size] [<image name>]
Takes an image filename and filter size (which must be an odd number) and returns blurred image based on filter size.
"""
__author__ = 'hdmjdp'
import numpy as np
import cv2
from math import sqrt


def readImage(filename):
    """
    Read in an image file, errors out if we can't find the file
    :param filename: The image filename.
    :return: The img object in matrix form.
    """
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        return img


def integralImage(img):
    """
    Returns the integral image/summed area table. See here: https://en.wikipedia.org/wiki/Summed_area_table
    :param img:
    :return:
    """
    height = img.shape[0]
    width = img.shape[1]
    int_image = np.zeros((height, width), np.uint64)
    for y in range(height):
        for x in range(width):
            up = 0 if (y - 1 < 0) else int_image.item((y - 1, x))
            left = 0 if (x - 1 < 0) else int_image.item((y, x - 1))
            diagonal = 0 if (x - 1 < 0 or y - 1 < 0) else int_image.item((y - 1, x - 1))
            val = img.item((y, x)) + int(up) + int(left) - int(diagonal)
            int_image.itemset((y, x), val)
    return int_image


def adjustEdges(height, width, point):
    """
    This handles the edge cases if the box's bounds are outside the image range based on current pixel.
    :param height: Height of the image.
    :param width: Width of the image.
    :param point: The current point.
    :return:
    """
    newPoint = [point[0], point[1]]
    if point[0] >= height:
        newPoint[0] = height - 1

    if point[1] >= width:
        newPoint[1] = width - 1
    return tuple(newPoint)


def findArea(int_img, a, b, c, d):
    """
    Finds the area for a particular square using the integral image. See summed area wiki.
    :param int_img: The
    :param a: Top left corner.
    :param b: Top right corner.
    :param c: Bottom left corner.
    :param d: Bottom right corner.
    :return: The integral image.
    """
    height = int_img.shape[0]
    width = int_img.shape[1]
    a = adjustEdges(height, width, a)
    b = adjustEdges(height, width, b)
    c = adjustEdges(height, width, c)
    d = adjustEdges(height, width, d)

    a = 0 if (a[0] < 0 or a[0] >= height) or (a[1] < 0 or a[1] >= width) else int_img.item(a[0], a[1])
    b = 0 if (b[0] < 0 or b[0] >= height) or (b[1] < 0 or b[1] >= width) else int_img.item(b[0], b[1])
    c = 0 if (c[0] < 0 or c[0] >= height) or (c[1] < 0 or c[1] >= width) else int_img.item(c[0], c[1])
    d = 0 if (d[0] < 0 or d[0] >= height) or (d[1] < 0 or d[1] >= width) else int_img.item(d[0], d[1])

    return a + d - b - c


def boxFilter(img, filterSize):
    """
    Runs the subsequent box filtering steps. Prints original image, finds integral image, and then outputs final image
    :param img: An image in matrix form.
    :param filterSize: The filter size of the matrix
    :return: A final image written as finalimage.png
    """
    # print("Printing original image...")
    # print(img)
    height = img.shape[0]
    width = img.shape[1]
    intImg = integralImage(img)
    finalImg = np.zeros((height, width), np.uint64)
    # print("Printing integral image...")
    # print(intImg)
    # cv2.imwrite("integral_image.png", intImg)
    loc = filterSize // 2
    for y in range(height):
        for x in range(width):
            finalImg.itemset((y, x), findArea(intImg, (y - loc - 1, x - loc - 1), (y - loc - 1, x + loc),
                                              (y + loc, x - loc - 1), (y + loc, x + loc)) / (filterSize ** 2))
    # print("Printing final image...")
    # # print(finalImg)
    #
    # cv2.imwrite("finalimage.png", finalImg)
    return finalImg


def boxFilterStd(img, filterSize):
    """
    Runs the subsequent box filtering steps. Prints original image, finds integral image, and then outputs final image
    :param img: An image in matrix form.
    :param filterSize: The filter size of the matrix
    :return: A final image written as finalimage.png
    """
    height = img.shape[0]
    width = img.shape[1]
    intImg_sum = integralImage(img)
    finalImg = np.zeros((height, width), np.uint64)
    img_square = np.square(img.astype(np.uint64))
    intImg_square = integralImage(img_square)
    loc = filterSize // 2
    for y in range(height):
        for x in range(width):
            s1 = findArea(intImg_square,  (y - loc - 1, x - loc - 1), (y - loc - 1, x + loc),
                                              (y + loc, x - loc - 1), (y + loc, x + loc))
            s2_sum = findArea(intImg_sum, (y - loc - 1, x - loc - 1), (y - loc - 1, x + loc),
                                              (y + loc, x - loc - 1), (y + loc, x + loc))
            s2 = s2_sum*s2_sum/(filterSize**2)
            variance = s1 - s2
            std_deviation = sqrt(variance/(filterSize**2))
            finalImg.itemset((y, x), std_deviation)
    # cv2.imwrite("finalimage11.png", finalImg)
    return finalImg


def boxFilter_MeanStd(img, filterSize):
    """
    Runs the subsequent box filtering steps. Prints original image, finds integral image, and then outputs final image
    :param img: An image in matrix form.
    :param filterSize: The filter size of the matrix
    :return: A final image written as finalimage.png
    """
    height = img.shape[0]
    width = img.shape[1]
    intImg_sum = integralImage(img)
    finalImg_mean = np.zeros((height, width), np.uint64)
    finalImg_stdv = np.zeros((height, width), np.uint64)

    img_square = np.square(img.astype(np.uint64))
    intImg_square = integralImage(img_square)
    loc = filterSize // 2
    for y in range(height):
        for x in range(width):
            s1 = findArea(intImg_square,  (y - loc - 1, x - loc - 1), (y - loc - 1, x + loc),
                                              (y + loc, x - loc - 1), (y + loc, x + loc))
            s2_sum = findArea(intImg_sum, (y - loc - 1, x - loc - 1), (y - loc - 1, x + loc),
                                              (y + loc, x - loc - 1), (y + loc, x + loc))
            mean = s2_sum / (filterSize ** 2)
            s2 = s2_sum * mean  # s2_sum*s2_sum/(filterSize**2)
            variance = s1 - s2
            std_deviation = sqrt(variance/(filterSize**2))
            finalImg_stdv.itemset((y, x), std_deviation)
            finalImg_mean.itemset((y, x), mean)
    # cv2.imwrite("finalImg_mean.png", finalImg_mean)
    # cv2.imwrite("finalImg_stdv.png", finalImg_stdv)
    return finalImg_mean, finalImg_stdv


def main():
    """
    Reads in image and handles argument parsing
    :return: None
    """

    img = readImage("datas/f3.jpg")
    src = cv2.resize(img, (400, 300), interpolation=cv2.INTER_AREA)
    # finalImg = boxFilter(src, 201)
    # finalImg = boxFilterStd(src, 201)
    finalImg_mean, finalImg_stdv = boxFilter_MeanStd(src, 201)
    cv2.imshow("finalImg", finalImg_mean)
    cv2.waitKey()


if __name__ == "__main__":
    main()