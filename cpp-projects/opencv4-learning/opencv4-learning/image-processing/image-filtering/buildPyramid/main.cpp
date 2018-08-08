#include <iostream>
#include <direct.h>
#include <stdlib.h>
#include <vector>
#include <string>
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
    vector<cv::Mat> dst;
    cv::buildPyramid(src,dst,4);
    vector<cv::Mat>::iterator it = dst.begin();
    int i = 1;
    for(;it!=dst.end();it++){
        stringstream title;
        title << "pyramid:" << i;
        cv::imshow(title.str(),*it);
        i++;
    }

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
