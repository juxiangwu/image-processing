#include "image-processing.h"
//#if defined(_MSC_VER) && (_MSC_VER >= 1600)
//#pragma execution_character_set("utf-8")
//#endif


//#include <iostream>
//#include <cmath>
//#include <limits.h>

//#include <cuda.h>

//#include <opencv2/opencv.hpp>


//using namespace cv;
//using namespace std;

//// cpu: 对图像进行缩放
//extern "C" void resizeImage(const Mat &src, Mat &dst, const Size &s);
//// cpu: 对图像进行平移  xlen左右   ylen上下
//extern "C" void transferImage(const Mat &src, Mat &dst, int xlen, int ylen);
//// cpu: 对图像镜面变换  num = 0 | 1
//extern "C" void mirrorImage(const Mat &src, Mat &dst, int num);
//// cpu: 对图像旋转变换
//extern "C" void rotateImage(const Mat &src, Mat &dst, int degree);
//// 对图像进行错切
//extern "C" void cutImage(const Mat &src, Mat &dst, int dir, int len);

// cuda 设备检测
//extern "C" bool initCUDA();

//extern "C" void resizeImageGPU(const Mat &_src, Mat &_dst, const Size &s);
//extern "C" void transferImageGPU(const Mat &_src, Mat &_dst, int xlen, int ylen);
//extern "C" void mirrorImageGPU(const Mat &_src, Mat &_dst, int num);
//extern "C" void rotateImageGPU(const Mat &src, Mat &dst, int degree);
//extern "C" void cutImageGPU(const Mat &_src, Mat &_dst, int dir, int len);


// cpu: 对图像进行缩放
extern "C"
void resizeImage(const Mat &src, Mat &dst, const Size &s)
{
    dst = Mat::zeros(s, CV_8UC3);
    double fRows = s.height / (float)src.rows;
    double fCols = s.width / (float)src.cols;
    int pX = 0;
    int pY = 0;
    for (int i = 0; i != dst.rows; ++i) {
        for (int j = 0; j != dst.cols; ++j) {
            pX = cvRound(i / (double)fRows);  // 四舍五入
            pY = cvRound(j / (double)fCols);
            if (pX < src.rows && pX >= 0 && pY < src.cols && pY >= 0) {
                dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(pX, pY)[0];  // B
                dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(pX, pY)[1];  // G
                dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(pX, pY)[2];  // R
            }
        }
    }
}

// cpu: 对图像进行平移  xlen左右   ylen上下
extern "C"
void transferImage(const Mat &src, Mat &dst, int xlen, int ylen)
{
    int width = src.cols, height = src.rows;
    width += abs(xlen);
    height += abs(ylen);

    dst = Mat::zeros(Size(width, height), CV_8UC3);

    int xadd = xlen < 0 ? 0 : abs(xlen);
    int yadd = ylen < 0 ? abs(ylen) : 0;

    for (int i = 0; i != src.rows; ++i) {
        for (int j = 0; j != src.cols; ++j) {
            dst.at<Vec3b>(i + yadd, j + xadd)[0] = src.at<Vec3b>(i, j)[0];
            dst.at<Vec3b>(i + yadd, j + xadd)[1] = src.at<Vec3b>(i, j)[1];
            dst.at<Vec3b>(i + yadd, j + xadd)[2] = src.at<Vec3b>(i, j)[2];
        }
    }
}

// cpu: 对图像镜面变换  num = 0 | 1;	(0:x轴；1:y轴)
extern "C"
void mirrorImage(const Mat &src, Mat &dst, int num)
{
    dst = Mat::zeros(Size(src.cols, src.rows), CV_8UC3);

    if (0 == num) {
        for (int i = 0, x = src.rows - 1; i != src.rows; ++i, --x) {
            for (int j = 0, y = 0; j != src.cols; ++j, ++y) {
                dst.at<Vec3b>(x, y)[0] = src.at<Vec3b>(i, j)[0];
                dst.at<Vec3b>(x, y)[1] = src.at<Vec3b>(i, j)[1];
                dst.at<Vec3b>(x, y)[2] = src.at<Vec3b>(i, j)[2];
            }
        }
    }
    else {
        for (int i = 0, x = 0; i != src.rows; ++i, ++x) {
            for (int j = 0, y = src.cols - 1; j != src.cols; ++j, --y) {
                dst.at<Vec3b>(x, y)[0] = src.at<Vec3b>(i, j)[0];
                dst.at<Vec3b>(x, y)[1] = src.at<Vec3b>(i, j)[1];
                dst.at<Vec3b>(x, y)[2] = src.at<Vec3b>(i, j)[2];
            }
        }
    }
}


// cpu: 对图像旋转变换    http://blog.csdn.net/ab1322583838/article/details/52102732    http://blog.csdn.net/fengbingchun/article/details/17713429
extern "C"
void rotateImage(const Mat &src, Mat &dst, int degree)
{
    degree = -degree;   // 原始为逆时针，取负转为顺时针
    double angle = degree * CV_PI / 180.;   // 转为弧度
    double a = sin(angle), b = cos(angle);
    int width = src.cols, height = src.rows;

    // 旋转后的新图尺寸
    int width_rotate = int(height * fabs(a) + width * fabs(b));
    int height_rotate = int(width * fabs(a) + height * fabs(b));

    dst = Mat::zeros(Size(width_rotate, height_rotate), CV_8UC3);

    // 旋转数组map
    // [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
    // [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
    float map[6];
    Mat map_matrix = Mat(2, 3, CV_32F, map);

    // 旋转中心
    CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
    CvMat map_matrix2 = map_matrix;
    cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);
    map[2] += (width_rotate - width) / 2;
    map[5] += (height_rotate - height) / 2;

    warpAffine(src, dst, map_matrix, Size(width_rotate, height_rotate), 0, 0, 0);      // 0,0,0 最近邻插值   1,0,0 双线性插值

                                                                                       //    imshow("cpu", dst);
}

// 对图像进行错切
extern "C"
void cutImage(const Mat &src, Mat &dst, int dir, int len)
{
    if (0 == dir) {
        dst = Mat(Size(src.cols + len, src.rows), CV_8UC3);

        uchar *src_data = src.data;
        uchar *dst_data = dst.data;

        double ratio = double(len) / double(dst.rows);

        for (int i = 0, x = 0; i < src.rows; i++, x++)
        {
            int start = (src.rows - i) * ratio;
            for (int j = start, y = 0; j < src.cols + start; j++, y++)
            {
                *(dst_data + (i*dst.cols + j) * 3 + 0) = *(src_data + (x*src.cols + y) * 3 + 0);
                *(dst_data + (i*dst.cols + j) * 3 + 1) = *(src_data + (x*src.cols + y) * 3 + 1);
                *(dst_data + (i*dst.cols + j) * 3 + 2) = *(src_data + (x*src.cols + y) * 3 + 2);
            }
        }
    }
    else {
        dst = Mat(Size(src.cols, src.rows + len), CV_8UC3);

        uchar *src_data = src.data;
        uchar *dst_data = dst.data;

        double ratio = double(len) / double(dst.cols);

        for (int j = 0, y = 0; j < src.cols; j++, y++)
        {
            int start = j * ratio;
            for (int i = start, x = 0; i < src.rows + start; i++, x++)
            {
                *(dst_data + (i*dst.cols + j) * 3 + 0) = *(src_data + (x*src.cols + y) * 3 + 0);
                *(dst_data + (i*dst.cols + j) * 3 + 1) = *(src_data + (x*src.cols + y) * 3 + 1);
                *(dst_data + (i*dst.cols + j) * 3 + 2) = *(src_data + (x*src.cols + y) * 3 + 2);
            }
        }
    }
}


//////////////////////////////////
// cuda 设备检测
extern "C"
bool initCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);
    return true;
}





////////////////////////////////
// gpu 缩放变换
extern "C"
__global__ void resizeKernel(uchar* _src_dev, uchar * _dst_dev, int _src_step, int _dst_step,
    int _src_rows, int _src_cols, int _dst_rows, int _dst_cols)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    double fRows = _dst_rows / (float)_src_rows;
    double fCols = _dst_cols / (float)_src_cols;

    int pX = 0;
    int pY = 0;

    pX = (int)(i / fRows);
    pY = (int)(j / fCols);
    if (pX < _src_rows && pX >= 0 && pY < _src_cols && pY >= 0) {
        *(_dst_dev + i*_dst_step + 3 * j + 0) = *(_src_dev + pX*_src_step + 3 * pY);
        *(_dst_dev + i*_dst_step + 3 * j + 1) = *(_src_dev + pX*_src_step + 3 * pY + 1);
        *(_dst_dev + i*_dst_step + 3 * j + 2) = *(_src_dev + pX*_src_step + 3 * pY + 2);

    }
}

extern "C"
void resizeImageGPU(const Mat &_src, Mat &_dst, const Size &s)
{
    _dst = Mat(s, CV_8UC3);
    uchar *src_data = _src.data;
    int width = _src.cols;
    int height = _src.rows;
    uchar *src_dev, *dst_dev;

    cudaMalloc((void**)&src_dev, 3 * width*height * sizeof(uchar));
    cudaMalloc((void**)&dst_dev, 3 * s.width * s.height * sizeof(uchar));
    cudaMemcpy(src_dev, src_data, 3 * width*height * sizeof(uchar), cudaMemcpyHostToDevice);

    int src_step = _src.step;   // 矩阵_src一行元素的字节数
    int dst_step = _dst.step;   // 矩阵_dst一行元素的字节数

    dim3 grid(s.height, s.width);
    resizeKernel<<< grid, 1 >>>(src_dev, dst_dev, src_step, dst_step, height, width, s.height, s.width);

    cudaMemcpy(_dst.data, dst_dev, 3 * s.width * s.height * sizeof(uchar), cudaMemcpyDeviceToHost);
}


////////////////////////////////
// gpu 平移变换
extern "C"
__global__ void transferKernel(uchar* _src_dev, uchar * _dst_dev, int width, int height,
    int _src_rows, int _src_cols, int xlen, int ylen)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    int xadd = xlen < 0 ? 0 : abs(xlen);
    int yadd = ylen < 0 ? abs(ylen) : 0;

    int offset = i*gridDim.y + j;
    int tran_offset = (i + yadd) * width + j + xadd;


    if (i < gridDim.x && j >= 0 && j < gridDim.y && i >= 0) {
        *(_dst_dev + tran_offset * 3 + 0) = *(_src_dev + offset * 3 + 0);
        *(_dst_dev + tran_offset * 3 + 1) = *(_src_dev + offset * 3 + 1);
        *(_dst_dev + tran_offset * 3 + 2) = *(_src_dev + offset * 3 + 2);
    }
}

extern "C"
void transferImageGPU(const Mat &_src, Mat &_dst, int xlen, int ylen)
{
    int width = _src.cols, height = _src.rows;
    width += abs(xlen);
    height += abs(ylen);

    _dst = Mat::zeros(Size(width, height), CV_8UC3);
    uchar *src_data = _src.data;
    uchar *src_dev, *dst_dev;

    cudaMalloc((void**)&src_dev, 3 * _src.rows * _src.cols * sizeof(uchar));
    cudaMalloc((void**)&dst_dev, 3 * width * height * sizeof(uchar));
    cudaMemcpy(src_dev, src_data, 3 * _src.rows * _src.cols * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemset(dst_dev, 0, 3 * width * height * sizeof(uchar));

    dim3 grid(_src.rows, _src.cols);
    //    cout << _src.rows << "  " << _src.cols << endl;
    transferKernel <<< grid, 1 >>>(src_dev, dst_dev, width, height, _src.rows, _src.cols, xlen, ylen);

    //    cout << width << "  " << height << "  " << _src.rows << "  " << _src.cols << endl;

    cudaMemcpy(_dst.data, dst_dev, 3 * width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
}


////////////////////////////////
// gpu 镜面变换
extern "C"
__global__ void mirrorKernel(uchar* _src_dev, uchar * _dst_dev, int height, int width, int num)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    int offset = i*gridDim.y + j;

    int x, y;

    if (0 == num) {
        x = height - i - 1;
        y = j;
    }
    else {
        x = i;
        y = width - j - 1;
    }

    int mirror_offset = x*gridDim.y + y;

    if (i < gridDim.x && j >= 0 && j < gridDim.y && i >= 0) {
        *(_dst_dev + mirror_offset * 3 + 0) = *(_src_dev + offset * 3 + 0);
        *(_dst_dev + mirror_offset * 3 + 1) = *(_src_dev + offset * 3 + 1);
        *(_dst_dev + mirror_offset * 3 + 2) = *(_src_dev + offset * 3 + 2);
    }
}

extern "C"
void mirrorImageGPU(const Mat &_src, Mat &_dst, int num)
{
    _dst = Mat::zeros(Size(_src.cols, _src.rows), CV_8UC3);
    uchar *src_data = _src.data;
    uchar *src_dev, *dst_dev;

    cudaMalloc((void**)&src_dev, 3 * _src.rows * _src.cols * sizeof(uchar));
    cudaMalloc((void**)&dst_dev, 3 * _src.rows * _src.cols * sizeof(uchar));
    cudaMemcpy(src_dev, src_data, 3 * _src.rows * _src.cols * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 grid(_src.rows, _src.cols);
    mirrorKernel <<< grid, 1 >>>(src_dev, dst_dev, _src.rows, _src.cols, num);

    cudaMemcpy(_dst.data, dst_dev, 3 * _src.rows * _src.cols * sizeof(uchar), cudaMemcpyDeviceToHost);
}


////////////////////////////////
// gpu 旋转变换
extern "C"
__device__ int saturateCast(double num)
{
    return round(num);
}

__global__ void rotateKernel(uchar* _src_dev, uchar * _dst_dev, int width, int height,
    const double m0, const double m1, const double m2, const double m3, const double m4, const double m5,
    int round_delta)
{
    int y = blockIdx.x;
    int x = blockIdx.y;

    //    if (y < gridDim.x && y > 0 && x < gridDim.y && x > 0)
    {

        int adelta = saturateCast(m0 * x * 1024);
        int bdelta = saturateCast(m3 * x * 1024);
        int X0 = saturateCast((m1 * y + m2) * 1024) + round_delta;
        int Y0 = saturateCast((m4 * y + m5) * 1024) + round_delta;
        int X = (X0 + adelta) >> 10;
        int Y = (Y0 + bdelta) >> 10;

        if ((unsigned)X < width && (unsigned)Y < height)
        {
            *(_dst_dev + (y*gridDim.y + x) * 3 + 0) = *(_src_dev + (Y*width + X) * 3 + 0);
            *(_dst_dev + (y*gridDim.y + x) * 3 + 1) = *(_src_dev + (Y*width + X) * 3 + 1);
            *(_dst_dev + (y*gridDim.y + x) * 3 + 2) = *(_src_dev + (Y*width + X) * 3 + 2);
        }
        else
        {
            *(_dst_dev + (y*gridDim.y + x) * 3 + 0) = 0;
            *(_dst_dev + (y*gridDim.y + x) * 3 + 1) = 0;
            *(_dst_dev + (y*gridDim.y + x) * 3 + 2) = 0;
        }

    }
}

extern "C"
void rotateImageGPU(const Mat &src, Mat &dst, int degree)
{
    degree = -degree;
    double angle = degree * CV_PI / 180.;
    double alpha = cos(angle);
    double beta = sin(angle);
    int width = src.cols;
    int height = src.rows;
    int width_rotate = cvRound(width * fabs(alpha) + height * fabs(beta));
    int height_rotate = cvRound(height * fabs(alpha) + width * fabs(beta));

    double m[6];
    m[0] = alpha;
    m[1] = beta;
    //    m[2] = (1 - alpha) * width / 2. - beta * height / 2.;
    m[2] = height * -beta;

    // cout << width << "   " << height << endl;
    // cout << width_rotate << "   " << height_rotate << endl;
    // cout << alpha << "      " << beta << endl;

    // cout << m[2] << endl;
    m[3] = -m[1];
    m[4] = m[0];
    //    m[5] = beta * width / 2. + (1 - alpha) * height / 2.;
    m[5] = 0;

    //    cout << "m[5]  " << m[5] << endl;

    Mat M = Mat(2, 3, CV_64F, m);
    dst = Mat(cv::Size(width_rotate, height_rotate), src.type(), cv::Scalar::all(0));

    double D = m[0] * m[4] - m[1] * m[3];
    D = D != 0 ? 1. / D : 0;
    double A11 = m[4] * D, A22 = m[0] * D;
    m[0] = A11; m[1] *= -D;
    m[3] *= -D; m[4] = A22;
    double b1 = -m[0] * m[2] - m[1] * m[5];
    double b2 = -m[3] * m[2] - m[4] * m[5];
    m[2] = b1; m[5] = b2;

    int round_delta = 512;  //  最近邻插值

                            // for (int y = 0; y < height_rotate; ++y)
                            // {
                            //     for (int x = 0; x < width_rotate; ++x)
                            //     {
                            //         int adelta = cv::saturate_cast<int>(m[0] * x * 1024);
                            //         int bdelta = cv::saturate_cast<int>(m[3] * x * 1024);
                            //         int X0 = cv::saturate_cast<int>((m[1] * y + m[2]) * 1024) + round_delta;
                            //         int Y0 = cv::saturate_cast<int>((m[4] * y + m[5]) * 1024) + round_delta;
                            //         int X = (X0 + adelta) >> 10;
                            //         int Y = (Y0 + bdelta) >> 10;

                            //         if ((unsigned)X < width && (unsigned)Y < height)
                            //         {
                            //       //      dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(Y, X);
                            //             *(dst.data + (y*width_rotate+x)*3 + 0) = *(src.data + (Y*width+X)*3 + 0);
                            //             *(dst.data + (y*width_rotate+x)*3 + 1) = *(src.data + (Y*width+X)*3 + 1);
                            //             *(dst.data + (y*width_rotate+x)*3 + 2) = *(src.data + (Y*width+X)*3 + 2);
                            //         }
                            //     }
                            // }

                            //    cout << saturate_cast<int>(-99999999999) << "   **" << endl;
                            //    cout << INT_MAX << endl;

    uchar *src_data = src.data;
    uchar *src_dev, *dst_dev;

    cudaMalloc((void**)&src_dev, 3 * src.rows * src.cols * sizeof(uchar));
    cudaMalloc((void**)&dst_dev, 3 * width_rotate * height_rotate * sizeof(uchar));
    cudaMemcpy(src_dev, src_data, 3 * src.rows * src.cols * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemset(dst_dev, 0, width_rotate * height_rotate * sizeof(uchar));

    dim3 grid(height_rotate, width_rotate);
    rotateKernel <<< grid, 1 >>>(src_dev, dst_dev, width, height,
        m[0], m[1], m[2], m[3], m[4], m[5], round_delta);

    cudaMemcpy(dst.data, dst_dev, 3 * width_rotate * height_rotate * sizeof(uchar), cudaMemcpyDeviceToHost);
}

////////////////////////////////
// gpu 错切变换
extern "C"
__global__ void cutKernel(uchar* _src_dev, uchar * _dst_dev, int width, double ratio, int dir)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int x = 0, y = 0;
    if (0 == dir)
    {
        y = (gridDim.x - i) * ratio;
    }
    else {
        x = j * ratio;
    }
    /*    int start = (gridDim.x - i) * ratio;
    int y = start;
    */
    int offset = i*gridDim.y + j;
    int tran_offset = (i + x) * width + j + y;

    if (i < gridDim.x && j >= 0 && j < gridDim.y && i >= 0) {
        *(_dst_dev + tran_offset * 3 + 0) = *(_src_dev + offset * 3 + 0);
        *(_dst_dev + tran_offset * 3 + 1) = *(_src_dev + offset * 3 + 1);
        *(_dst_dev + tran_offset * 3 + 2) = *(_src_dev + offset * 3 + 2);
    }
}

/*__global__ void cutKernel1(uchar* _src_dev, uchar * _dst_dev, int width, double ratio)
{
int i = blockIdx.x;
int j = blockIdx.y;
int start = j * ratio;
int x = start;
int offset = i*gridDim.y + j;
int tran_offset = (i+x) * width + j;
if (i < gridDim.x && j >= 0 && j < gridDim.y && i >= 0)  {
*(_dst_dev + tran_offset*3 + 0) = *(_src_dev + offset*3 + 0);
*(_dst_dev + tran_offset*3 + 1) = *(_src_dev + offset*3 + 1);
*(_dst_dev + tran_offset*3 + 2) = *(_src_dev + offset*3 + 2);
}
}*/

extern "C"
void cutImageGPU(const Mat &_src, Mat &_dst, int dir, int len)
{
    int width = _src.cols, height = _src.rows;

    /*    if (0 == dir) {
    width += len;
    _dst = Mat::zeros(Size(width, height), CV_8UC3);
    uchar *src_data = _src.data;
    uchar *src_dev , *dst_dev;
    cudaMalloc( (void**)&src_dev, 3 * _src.rows * _src.cols * sizeof(uchar) );
    cudaMalloc( (void**)&dst_dev, 3 * width * height * sizeof(uchar) );
    cudaMemcpy(src_dev, src_data, 3 * _src.rows * _src.cols * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemset(dst_dev, 0, 3 * width * height * sizeof(uchar));
    double ratio = (double)len / _dst.rows;
    dim3 grid(_src.rows, _src.cols);
    cutKernel <<< grid, 1 >>>(src_dev, dst_dev, width, ratio, dir);
    cudaMemcpy(_dst.data, dst_dev, 3 * width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
    } else {
    height += len;
    _dst = Mat::zeros(Size(width, height), CV_8UC3);
    uchar *src_data = _src.data;
    uchar *src_dev , *dst_dev;
    cudaMalloc( (void**)&src_dev, 3 * _src.rows * _src.cols * sizeof(uchar) );
    cudaMalloc( (void**)&dst_dev, 3 * width * height * sizeof(uchar) );
    cudaMemcpy(src_dev, src_data, 3 * _src.rows * _src.cols * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemset(dst_dev, 0, 3 * width * height * sizeof(uchar));
    double ratio = (double)len / _dst.cols;
    dim3 grid(_src.rows, _src.cols);
    cutKernel1 <<< grid, 1 >>>(src_dev, dst_dev, width, ratio, dir);
    cudaMemcpy(_dst.data, dst_dev, 3 * width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
    }*/
    double ratio;
    if (0 == dir) {
        width += len;
        ratio = (double)len / height;
    }
    else {
        height += len;
        ratio = (double)len / width;
    }

    _dst = Mat::zeros(Size(width, height), CV_8UC3);
    uchar *src_data = _src.data;
    uchar *src_dev, *dst_dev;

    cudaMalloc((void**)&src_dev, 3 * _src.rows * _src.cols * sizeof(uchar));
    cudaMalloc((void**)&dst_dev, 3 * width * height * sizeof(uchar));
    cudaMemcpy(src_dev, src_data, 3 * _src.rows * _src.cols * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemset(dst_dev, 0, 3 * width * height * sizeof(uchar));


    dim3 grid(_src.rows, _src.cols);
    cutKernel <<< grid, 1 >>>(src_dev, dst_dev, width, ratio, dir);

    cudaMemcpy(_dst.data, dst_dev, 3 * width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
}

//int main()
//{
//    Mat src = cv::imread("f.bmp" , 1);      // 读入图片


//    Mat dst_scale_cpu;
//    Mat dst_scale_gpu;

//    Mat dst_trans_cpu;
//    Mat dst_trans_gpu;

//    Mat dst_mirror_cpu;
//    Mat dst_mirror_gpu;

//    Mat dst_rotate_cpu;
//    Mat dst_rotate_gpu;

//    Mat dst_cut_cpu;
//    Mat dst_cut_gpu;

///*
//    struct timeval start;
//    struct timeval end;
//    unsigned long timer;

//    gettimeofday(&start, NULL);     // 开始计时
//    resizeImage(src, dst_scale_cpu, Size(src.cols * 2, src.rows * 2));        // CPU 图片缩放   缩放后的结果存放在dst_cpu中    第三个参数为缩放大小
//    gettimeofday(&end, NULL);       // 结束计时
//    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "cpu缩放所耗费的时间：" << timer << "us\n";
//*/

//    struct timeval start;
//    struct timeval end;
//    unsigned long timer;
//    gettimeofday(&start, NULL);     // 开始计时
/////////////////////////////
//    resizeImage(src, dst_scale_cpu, Size(src.cols * 2, src.rows * 2));
//    transferImage(src, dst_trans_cpu, 100, -100);
//    mirrorImage(src, dst_mirror_cpu, 1);
//    rotateImage(src, dst_rotate_cpu, 30);
//    cutImage(src, dst_cut_cpu, 0, 50);
/////////////////////////////
//    gettimeofday(&end, NULL);       // 结束计时
//    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "cpu所耗费的时间：" << timer << "us\n";





//    initCUDA();





//    gettimeofday(&start, NULL);
/////////////////////////////
//    resizeImageGPU(src, dst_scale_gpu, Size(src.cols * 2, src.rows * 2));
//    transferImageGPU(src, dst_trans_gpu, 100, -100);
//    mirrorImageGPU(src, dst_mirror_gpu, 1);
//    rotateImageGPU(src, dst_rotate_gpu, 30);
//    cutImageGPU(src, dst_cut_gpu, 0, 50);
/////////////////////////////
//    gettimeofday(&end, NULL);
//    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "gpu所耗费的时间：" << timer << "us\n";




////////////////////////////
//    imshow("原图", src);
//    imshow("缩放_cpu", dst_scale_cpu);
//    imshow("缩放_gpu", dst_scale_gpu);
//    imshow("平移_cpu", dst_trans_cpu);
//    imshow("平移_gpu", dst_trans_gpu);
//    imshow("镜像_cpu", dst_mirror_cpu);
//    imshow("镜像_gpu", dst_mirror_gpu);
//    imshow("旋转_cpu", dst_rotate_cpu);
//    imshow("旋转_gpu", dst_rotate_gpu);
//    imshow("错切_cpu", dst_cut_cpu);
//    imshow("错切_gpu", dst_cut_gpu);



//    // transferImage(src, dst_trans_cpu, 100, -100);
//    // imshow("cpu_trans", dst_trans_cpu);
//    // transferImageGPU(src, dst_trans_gpu, 100, -100);
//    // imshow("gpu_trans", dst_trans_gpu);

//    // mirrorImage(src, dst_mirror_cpu, 1);
//    // mirrorImageGPU(src, dst_mirror_gpu, 1);
//    // imshow("gpu", dst_mirror_gpu);


//    // rotateImage(src, dst_rotate_cpu, 30);
//    // rotateImageGPU(src, dst_rotate_gpu, 30);
//    // imshow("gpu", dst_rotate_gpu);


//    // cutImage(src, dst_cut_cpu, 0, 50);
//    // imshow("cpu", dst_cut_cpu);
//    // cutImageGPU(src, dst_cut_gpu, 0, 50);
//    // imshow("gpu", dst_cut_gpu);
///*
//    initCUDA();
//    Mat dst_gpu;

//    gettimeofday(&start, NULL);
//    resizeImageGPU(src, dst_gpu, Size(src.cols * 2, src.rows * 2));

////    imshow("src", src);

////    imshow(" ", dst_gpu);

//    gettimeofday(&end, NULL);
//    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "gpu缩放所耗费的时间：" << timer << "us\n";

////    imshow("Demo", dst_gpu);

//*/

//    waitKey(0);

//    return 0;
//}
