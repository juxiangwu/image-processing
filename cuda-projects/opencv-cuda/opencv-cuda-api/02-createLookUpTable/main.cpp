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

cv::Mat generateLookupTable(){
    cv::Mat lookupTable(1,256,CV_8U);
    uchar *p = lookupTable.data;

    for(int i = 0;i < 256;i++){
        p[i] = 255 - i;
    }
    return lookupTable;
}

int main()
{
    try
    {
        cv::Mat host_src = cv::imread("../../../../datas/bird.jpg");
        cv::cuda::GpuMat dev_src;
        dev_src.upload(host_src);
        cv::Mat host_dst;

        cv::Mat hostLookupTable = generateLookupTable();
        cv::Ptr<cv::cuda::LookUpTable> devLookupTable = cv::cuda::createLookUpTable(hostLookupTable);
        //执行LUT并将数据从CUDA设备复制到HOST
        devLookupTable->transform(dev_src,host_dst);

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
