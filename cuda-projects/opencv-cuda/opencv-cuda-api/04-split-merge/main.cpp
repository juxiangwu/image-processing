#include <iostream>
#include <vector>
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

        std::vector<cv::cuda::GpuMat> dev_channels;

        //分离图像通道
        cv::cuda::split(dev_src,dev_channels);

        cv::Mat host_b,host_g,host_r;
         //复制CUDA数据到HOST
        if(dev_channels.size() == 3){
            dev_channels[0].download(host_b);
            dev_channels[1].download(host_g);
            dev_channels[2].download(host_r);

            cv::imshow("channel:R",host_r);
            cv::imshow("channel:G",host_g);
            cv::imshow("channel:B",host_b);
        }


        cv::cuda::merge(dev_channels,host_dst);
//        dev_dst.download(host_dst);

        cv::imshow("SRC",host_src);
        cv::imshow("Result:Merge",host_dst);
        cv::waitKey();
        cv::destroyAllWindows();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
