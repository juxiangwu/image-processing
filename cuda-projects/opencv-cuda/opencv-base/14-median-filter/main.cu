#include <iostream>
#include "opencv2/opencv.hpp"

int main ()
{
    cv::Mat h_img1 = cv::imread("../../../../datas/cameraman.tif",0);
    cv::cuda::GpuMat h_src,h_result;
    h_src.upload(h_img1);
    cv::Ptr<cv::cuda::Filter> filter7x7;
    filter7x7 = cv::cuda::createMedianFilter(CV_8UC1,7);
    filter7x7->apply(h_src,h_result);

    cv::Mat result;
    h_result.download(result);
//    cv::medianBlur(h_img1,h_result,3);
    cv::imshow("Original Image ", h_img1);
    cv::imshow("Median Blur Result", result);
    cv::waitKey();
    return 0;
}
