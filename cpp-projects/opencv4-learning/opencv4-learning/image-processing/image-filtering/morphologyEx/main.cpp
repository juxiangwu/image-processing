#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    cv::Mat src = cv::imread("../../../../../../datas/char.jpg");
    if(src.empty()){
        std::cerr << "cannot open image" << std::endl;
        return -1;
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));

    cv::Mat dst_dilate;
    cv::morphologyEx(src,dst_dilate,cv::MORPH_DILATE,kernel);
    cv::imshow("dst_dilate",dst_dilate);

    cv::Mat dst_erode;
    cv::morphologyEx(src,dst_erode,cv::MORPH_ERODE,kernel);
    cv::imshow("dst_erode",dst_erode);

    cv::Mat dst_close;
    cv::morphologyEx(src,dst_close,cv::MORPH_CLOSE,kernel);
    cv::imshow("dst_close",dst_close);

    cv::Mat dst_open;
    cv::morphologyEx(src,dst_open,cv::MORPH_OPEN,kernel);
    cv::imshow("dst_open",dst_open);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
