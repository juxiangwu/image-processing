
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

//出错处理函数
#define CHECK_ERROR(call){\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}

//内核函数：实现上下翻转
__global__ void swap_image_kernel(cuda::PtrStepSz<uchar3> cu_src, cuda::PtrStepSz<uchar3> cu_dst, int h, int w)
{
	//计算的方法：参看前面两文
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	//为啥要这样限制：参看前面两文
	if (x < cu_src.cols && y < cu_src.rows)
	{
		//为何不是h-y-1,而不是h-y，自己思考哦
		cu_dst(y, x) = cu_src(h - y - 1, x);
	}
}
//调用函数，主要处理block和grid的关系
void swap_image(cuda::GpuMat src, cuda::GpuMat dst, int h, int w)
{
	assert(src.cols == w && src.rows == h);
	int uint = 32;
	//参考前面两文的block和grid的计算方法，注意不要超过GPU限制
	dim3 block(uint, uint);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	printf("grid = %4d %4d %4d\n", grid.x, grid.y, grid.z);
	printf("block= %4d %4d %4d\n", block.x, block.y, block.z);
	swap_image_kernel << <grid, block >> > (src, dst, h, w);
	//同步一下，因为计算量可能很大
	CHECK_ERROR(cudaDeviceSynchronize());
}
int main(int argc, char **argv)
{
	Mat src, dst;
	cuda::GpuMat cu_src, cu_dst;
	int h, w;
	//根据argv[1]读入图片数据，BGR格式读进来
	src = imread(argv[1]);
	//检测是否正确读入
	if (src.data == NULL)
	{
		cout << "Read image error" << endl;
		return -1;
	}
	h = src.rows; w = src.cols;
	cout << "图片高：" << h << ",图片宽：" << w << endl;
	//上传CPU图像数据到GPU，跟cudaMalloc和cudaMemcpy很像哦，其实upload里面就是这么写的
	cu_src.upload(src);
	//申请GPU空间，也可以到函数里申请，不管怎样总要申请，要不然内核函数会爆掉哦
	cu_dst = cuda::GpuMat(h, w, CV_8UC3, Scalar(0, 0, 0));
	//申请CPU空间
	dst = Mat(h, w, CV_8UC3, Scalar(0, 0, 0));
	//调用函数swap_image,由该函数调用内核函数，这样层次分明，不容易出错
	//当然你也可以直接在这里调用内核函数，东西太多代码容易乱
	swap_image(cu_src, cu_dst, h, w);
	//下载GPU数据到CPU，与upload()对应
	cu_dst.download(dst);
	//显示cpu图像，如果安装了openCV集成了openGL,那可以直接显示GpuMat
	imshow("dst", dst);
	//等待按键
	waitKey();
	//写图片到文件
	if (argc == 3)
		imwrite(argv[2], dst);
	return 0;
}
