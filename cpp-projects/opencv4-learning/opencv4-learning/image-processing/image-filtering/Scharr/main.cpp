#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    cv::Mat src = cv::imread("../../../../../../datas/f4.jpg");
    if(src.empty()){
        std::cerr << "cannot open image" << std::endl;
        return -1;
    }
    cv::Mat dst_x,dst_y,dst;

    cv::Scharr(src,dst_x,CV_16S,1,0);
    cv::Scharr(src,dst_y,CV_16S,0,1);

    cv::addWeighted(dst_x,0.5,dst_y,0.5,0,dst);

    cv::imshow("src",src);
    cv::imshow("dst",dst);
    cv::imshow("dst_x",dst_x);
    cv::imshow("dst_y",dst_y);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
