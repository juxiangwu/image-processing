#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#define CVUI_IMPLEMENTATION
#include <cvui.h>
extern "C"{
#include <uv.h>
}

using namespace std;
#define WINDOW_NAME	"CUDA Magnitude"
cv::Mat src1,dst,roi1,roi2,dst_roi,frame;

void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 120, 200, "Settings");
    cv::Mat src1_rz;
    cv::resize(src1,src1_rz,cv::Size(src1.cols/2,src1.rows/2));

    src1_rz.copyTo(roi1);

    if (cvui::button(frame, 15, 50,100,30, "magnitude xy")) {
        cv::Mat gray;
        cv::cvtColor(src1,gray,cv::COLOR_BGR2GRAY);
        cv::Mat dx,dy;
        cv::Sobel(gray,dx,-1,1,0);
        cv::Sobel(gray,dy,-1,0,1);
        dx.convertTo(dx,CV_32F);
        dy.convertTo(dy,CV_32F);
        cv::cuda::magnitude(dx,dy,dst);
        dst.convertTo(dst,CV_8U);
        cv::normalize(dst,dst,0,255,cv::NORM_MINMAX);
        cv::convertScaleAbs(dst,dst);
        cv::cvtColor(dst,dst,cv::COLOR_GRAY2BGR);
    }

    if (cvui::button(frame, 15, 90,100,30, "magnitudeSqr")) {
        cv::Mat gray;
        cv::cvtColor(src1,gray,cv::COLOR_BGR2GRAY);
        cv::Mat dx,dy;
        cv::Sobel(gray,dx,-1,1,0);
        cv::Sobel(gray,dy,-1,0,1);
        dx.convertTo(dx,CV_32F);
        dy.convertTo(dy,CV_32F);
        cv::cuda::magnitudeSqr(dx,dy,dst);
        dst.convertTo(dst,CV_8U);
        cv::normalize(dst,dst,0,255,cv::NORM_MINMAX);
        cv::convertScaleAbs(dst,dst);
        cv::cvtColor(dst,dst,cv::COLOR_GRAY2BGR);
    }

    if(!dst.empty()){
        cv::Mat dst_rz;
        cv::resize(dst,dst_rz,cv::Size(dst.cols / 2,dst.rows/2));
        dst_rz.copyTo(dst_roi);
    }

    cvui::update();

    // Show everything on the screen
    cv::imshow(WINDOW_NAME, frame);

    // Check if ESC was pressed
    if (cv::waitKey(30) == 27) {
        uv_idle_stop(handle);
    }

}

int main()
{
    src1 = cv::imread("../../../../datas/f1.jpg");
    frame = cv::Mat(600,1024,CV_8UC3);
    roi1 = frame(cv::Rect(180,10,src1.cols / 2,src1.rows / 2));
    std::cout<<"roi1:" << cv::Rect(100,10,src1.cols / 2,src1.rows / 2) << std::endl;

    dst_roi = frame(cv::Rect(190+src1.cols / 2,10,src1.cols / 2,src1.rows / 2));
    cvui::init(WINDOW_NAME);
    uv_idle_t idler;
    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);

    uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    cv::destroyAllWindows();
    return 0;

}
