#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
//https://blog.csdn.net/jia20003/article/details/69802932
int main()
{
    cv::VideoCapture cap(1);
    if(!cap.isOpened()){
        std::cerr << "cannot open camera" << std::endl;
        return -1;
    }
    cv::UMat frame,gray;
    while(true){
        cap >> frame;
        if(frame.empty()){
            continue;
        }
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
        cv::imshow("UMat Demo",gray);
        char key = cv::waitKey(10);
        if(key == 27){
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
