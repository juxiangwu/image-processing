#include <iostream>
#include <opencv2/opencv.hpp>
#include "similiar_process.h"

using namespace std;

int main()
{
    cv::Mat src = cv::imread("d:/develop/dl/projects/resources/images/nude-1.jpg",0);
    cv::Mat dst;
    FacePreprocess::similarTransform(src,dst);
    cv::imshow("dst",dst);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
