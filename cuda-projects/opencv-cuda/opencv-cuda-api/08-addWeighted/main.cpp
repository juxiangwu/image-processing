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
        cv::Mat host_src_1 = cv::imread("../../../../datas/f1.jpg");
        cv::Mat host_src_2 = cv::imread("../../../../datas/f2.jpg");

        cv::Mat host_dst;
        cv::cuda::addWeighted(host_src_1,0.45,
                              host_src_2,0.55,2.4,
                              host_dst);

        cv::imshow("SRC_1",host_src_1);
        cv::imshow("SRC_2",host_src_2);
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
