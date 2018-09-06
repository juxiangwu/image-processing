#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H
#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H


#include <iostream>
#include <cmath>
#include <limits.h>

#include <cuda.h>

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

// cpu: 对图像进行缩放
extern "C" void resizeImage(const Mat &src, Mat &dst, const Size &s);
// cpu: 对图像进行平移  xlen左右   ylen上下
extern "C" void transferImage(const Mat &src, Mat &dst, int xlen, int ylen);
// cpu: 对图像镜面变换  num = 0 | 1
extern "C" void mirrorImage(const Mat &src, Mat &dst, int num);
// cpu: 对图像旋转变换
extern "C" void rotateImage(const Mat &src, Mat &dst, int degree);
// 对图像进行错切
extern "C" void cutImage(const Mat &src, Mat &dst, int dir, int len);

// cuda 设备检测
extern "C" bool initCUDA();

extern "C" void resizeImageGPU(const Mat &_src, Mat &_dst, const Size &s);
extern "C" void transferImageGPU(const Mat &_src, Mat &_dst, int xlen, int ylen);
extern "C" void mirrorImageGPU(const Mat &_src, Mat &_dst, int num);
extern "C" void rotateImageGPU(const Mat &src, Mat &dst, int degree);
extern "C" void cutImageGPU(const Mat &_src, Mat &_dst, int dir, int len);

#endif // IMAGEPROCESS_H
#endif // IMAGEPROCESSING_H
