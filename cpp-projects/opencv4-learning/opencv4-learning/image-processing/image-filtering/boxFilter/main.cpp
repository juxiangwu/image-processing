#include <iostream>
#include <direct.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
#define MAX_PATH 260
int main()
{
    cv::Mat src = cv::imread("../../../../../../datas/f4.jpg");
    if(src.empty()){
        std::cerr << "cannot open image" << std::endl;
        return -1;
    }
    cv::Mat dst;
    cv::boxFilter(src,dst,-1,cv::Size(9,9));

    cv::imshow("src",src);
    cv::imshow("dst",dst);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
