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
    cv::Mat dst_gaussian;
    cv::Mat dst_gobar;
//    cv::Mat dst_deriv_x,dst_deriv_y;

    cv::Mat kernel_gaussian = cv::getGaussianKernel(5,2.4);
    cv::Mat kernel_gabor = cv::getGaborKernel(cv::Size(5,5),2.4,10,15,2.4);
//    cv::Mat kernel_deriv_x,kernel_deriv_y;
//    cv::getDerivKernels(kernel_deriv_x,kernel_deriv_y,3,3,3);

    cv::filter2D(src,dst_gaussian,-1,kernel_gaussian);
    cv::filter2D(src,dst_gobar,-1,kernel_gabor);
//    cv::filter2D(src,dst_deriv_x,-1,kernel_deriv_x);
//    cv::filter2D(src,dst_deriv_y,-1,kernel_deriv_y);

    cv::imshow("src",src);
    cv::imshow("dst_gaussian",dst_gaussian);
    cv::imshow("dst_gabor",dst_gobar);
//    cv::imshow("dst_deriv_x",dst_deriv_x);
//    cv::imshow("dst_deriv_y",dst_deriv_y);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
