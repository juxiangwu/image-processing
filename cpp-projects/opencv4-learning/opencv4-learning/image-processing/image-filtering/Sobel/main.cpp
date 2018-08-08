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
    cv::Mat dst,dst_x,dst_y;

    cv::Sobel(src,dst_x,-1,1,0);
    cv::Sobel(src,dst_y,-1,0,1);

    cv::addWeighted(dst_x,0.5,dst_y,0.5,0,dst);

    cv::imshow("src",src);
    cv::imshow("dst",dst);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
