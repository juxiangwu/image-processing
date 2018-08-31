#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
void rotate2D(const cv::Mat & src, cv::Mat & dst, const double degrees)
{
    cv::Point2f center(src.cols/2.0, src.rows/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, degrees, 1.0);
    cv::Rect bbox = cv::RotatedRect(center,src.size(), degrees).boundingRect();

    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;

    cv::warpAffine(src, dst, rot, bbox.size());
}

int main()
{
    const double degrees = 45;

    cv::Mat src = cv::imread("d:/develop/dl/projects/resources/images/dog-cycle-car.png", cv::IMREAD_UNCHANGED);
    cv::Mat dst;

    rotate2D(src, dst, degrees);
    cv::imshow("dst",dst);
    cv::waitKey();
    cv::destroyAllWindows();
    //cv::imwrite("frog_rotated.png", dst);
    return 0;
}
