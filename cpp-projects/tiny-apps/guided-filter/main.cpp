#include <iostream>
#include "guidedfilter.h"
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    cv::Mat I = cv::imread("../../../datas/cat.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat p = I;

    int r = 4; // try r=2, 4, or 8
    double eps = 0.2 * 0.2; // try eps=0.1^2, 0.2^2, 0.4^2

    eps *= 255 * 255;   // Because the intensity range of our images is [0, 255]

    cv::Mat q = guidedFilter(I, p, r, eps);
    cv::imshow("src",I);
    cv::imshow("dst",p);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
