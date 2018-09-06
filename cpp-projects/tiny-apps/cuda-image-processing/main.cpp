#include <iostream>
#include "image-processing.h"
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    bool res = initCUDA();
    std::cout << "init cuda:"<<res << std::endl;
    if(!res){
        std::cerr << "cannot init cuda device.\n";
        return -1;
    }
    cv::VideoCapture cap(0);
    if(!cap.isOpened()){
        std::cerr << "cannot open camera.\n";
        return -1;
    }
    cv::Mat frame;
    cv::Mat dst;
    while(true){
        cap >> frame;
        if(frame.empty()){
            continue;
        }
        rotateImageGPU(frame,dst,45);
        cv::imshow("src",frame);
        cv::imshow("dst",dst);
        char key = cv::waitKey(10);
        if(key == 27){
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
