#include <iostream>
#include <opencv2/opencv.hpp>
#include "zhangsuen.h"

using namespace std;

int main()
{
    cv::Mat src = cv::imread("d:/develop/dl/projects/resources/images/char.jpg",0);
    thin(src,false,false,true);
    cv::imshow("src",src);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
