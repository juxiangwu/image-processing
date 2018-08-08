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
    cv::Mat dst;
    cv::blur(src,dst,cv::Size(11,11));

    cv::imshow("src",src);
    cv::imshow("dst",dst);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
