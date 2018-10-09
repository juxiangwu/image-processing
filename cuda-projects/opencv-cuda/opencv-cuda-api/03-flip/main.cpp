#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>


using namespace std;


int main()
{
    try
    {
        cv::Mat host_src = cv::imread("../../../../datas/bird.jpg");
        cv::cuda::GpuMat dev_src,dev_dst;

        //复制HOST数据到CUDA设备
        dev_src.upload(host_src);
        cv::Mat host_dst;

        cv::cuda::flip(dev_src,dev_dst,1);

        //复制CUDA数据到HOST
        dev_dst.download(host_dst);

        cv::imshow("SRC",host_src);

        cv::imshow("Result", host_dst);
        cv::waitKey();
        cv::destroyAllWindows();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
