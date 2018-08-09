#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;

int main()
{
    cv::Mat src = cv::imread("../../../../../../datas/f4.jpg");
    if(src.empty()){
        std::cerr << "cannot open image" << std::endl;
        return -1;
    }
    cv::Mat dst_linear;
    cv::Point2f center(src.cols/2,src.rows/2);
    double maxRadius = 0.7*min(center.y, center.x);

    int flags = cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS;
    cv::linearPolar(src,dst_linear,center,maxRadius,flags);

    double M = src.rows / log(maxRadius);
    cv::Mat dst_log;
    cv::logPolar(src,dst_log,center,M,flags);

    cv::imshow("src",src);
    cv::imshow("dst_linear_polar",dst_linear);
    cv::imshow("dst_log_polar",dst_log);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
