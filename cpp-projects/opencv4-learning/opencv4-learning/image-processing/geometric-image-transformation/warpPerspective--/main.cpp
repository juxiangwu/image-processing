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
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];

    //设置三个点来计算仿射变换
    srcTri[0] = cv::Point2f(0, 0);
    srcTri[1] = cv::Point2f(src.cols - 1, 0);
    srcTri[2] = cv::Point2f(0, src.rows - 1);

    dstTri[0] = cv::Point2f(src.cols*0.0, src.rows*0.33);
    dstTri[1] = cv::Point2f(src.cols*0.85, src.rows*0.25);
    dstTri[2] = cv::Point2f(src.cols*0.15, src.rows*0.7);
    cv::Mat M = cv::getAffineTransform(srcTri,dstTri);
    cv::Mat dst_affine = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::warpAffine(src,dst_affine,M,dst_affine.size());

    //计算旋转
    cv::Point2f center(src.rows / 2,src.cols / 2);
    cv::Mat rotate_M = cv::getRotationMatrix2D(center,30,1.0);
    cv::Mat dst_rotate = cv::Mat::zeros(src.rows,src.cols,src.type());
    cv::warpAffine(src,dst_rotate,rotate_M,dst_rotate.size());

    cv::imshow("src",src);
    cv::imshow("dst_affine",dst_affine);
    cv::imshow("dst_rotate",dst_rotate);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
