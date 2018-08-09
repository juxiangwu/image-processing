#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * 图像去畸变
 * @brief undistort
 * @param frame
 */

void undistort(cv::Mat frame) {

    float height = frame.rows;
    float width = frame.cols;

    cv::Mat frame1;
    frame.copyTo(frame1);
    frame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
    cv::Mat mapx, mapy, mapX, mapY;
    cv::Mat cM = (cv::Mat_<double>(3,3) << 6.2595267531072898e+02/640*width, 0, width/2, 0, 6.2595267531072898e+02/480*height, height/2, 0, 0, 1);
    cv::Mat par =  (cv::Mat_<double>(1,5) << 1.8089925314920607e-01, -1.1010298405973520e+00, 0, 0, 1.4124526045744463e+00);
    cv::Mat newM;

    cv::initUndistortRectifyMap(cM, par, cv::Mat_<double>::eye(3,3), newM, frame.size(), CV_32FC1, mapx, mapy);
    cv::convertMaps(mapx, mapy, mapX, mapY, CV_16SC2, false);//to change on true
    cv::remap(frame1, frame, mapX, mapY, cv::INTER_LINEAR);
}

int main()
{

    cv::Mat src = cv::imread("../../../../../../../datas/undistort.jpg");
    if(src.empty()){
        std::cerr << "cannot open image" << std::endl;
        return -1;
    }
    undistort(src);
    cv::imshow("result",src);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
