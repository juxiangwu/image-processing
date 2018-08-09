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
    int img_height = src.rows;
    int img_width = src.cols;
    vector<cv::Point2f> corners(4);
    corners[0] = cv::Point2f(0,0);
    corners[1] = cv::Point2f(img_width-1,0);
    corners[2] = cv::Point2f(0,img_height-1);
    corners[3] = cv::Point2f(img_width-1,img_height-1);
    vector<cv::Point2f> corners_trans(4);
    corners_trans[0] = cv::Point2f(150,250);
    corners_trans[1] = cv::Point2f(771,0);
    corners_trans[2] = cv::Point2f(0,img_height-1);
    corners_trans[3] = cv::Point2f(650,img_height-1);

    cv::Mat transform = cv::getPerspectiveTransform(corners,corners_trans);
    cout<<transform<<endl;
    vector<cv::Point2f> ponits, points_trans;
    for(int i=0;i<img_height;i++){
        for(int j=0;j<img_width;j++){
            ponits.push_back(Point2f(j,i));
        }
    }

    cv::perspectiveTransform( ponits, points_trans, transform);
    cv::Mat img_trans = cv::Mat::zeros(img_height,img_width,CV_8UC3);
    int count = 0;
    for(int i=0;i<img_height;i++){
        uchar* p = img.ptr<uchar>(i);
        for(int j=0;j<img_width;j++){
            int y = points_trans[count].y;
            int x = points_trans[count].x;
            uchar* t = img_trans.ptr<uchar>(y);
            t[x*3]  = p[j*3];
            t[x*3+1]  = p[j*3+1];
            t[x*3+2]  = p[j*3+2];
            count++;
        }
    }
    return 0;
}
