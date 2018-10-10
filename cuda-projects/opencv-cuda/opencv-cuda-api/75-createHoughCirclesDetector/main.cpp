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
#define WINDOW_NAME	"CUDA createHoughCirclesDetector"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame;
cv::Mat src1_rz;
int m_type = 1;
int setting_width = 200,setting_height=300;
int offset = 10;
cv::Mat dst_rz;
cv::Ptr<cv::cuda::HoughCirclesDetector> hough;

void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 200, 300, "Settings");
    //    src2.copyTo(roi2);
    src1_rz.copyTo(roi1);

    if (cvui::button(frame, 20, 230,180,30, "createHoughCirclesDetector")) {
        cv::Mat gray;
        cv::cvtColor(src1,gray,cv::COLOR_BGR2GRAY);
        cv::cuda::GpuMat dev_src(gray),dev_dst,dev_dst_circles;
        cv::cuda::threshold(dev_src,dev_dst,180,240,cv::THRESH_BINARY);
        vector<cv::Vec3f> circles;

        hough = cv::cuda::createHoughCirclesDetector(1, 100, 122, 50, 1,
                                                     max(src1.size().width, src1.size().height));
        hough->detect(dev_dst,dev_dst_circles);
        circles.resize(dev_dst_circles.size().width);
        src1.copyTo(dst);
        if (!circles.empty())
            dev_dst_circles.row(0).download(cv::Mat(circles).reshape(3, 1));
        for( size_t i = 0; i < circles.size(); i++ )
        {
            cv::Vec3i cir = circles[i];
            cv::circle( dst, cv::Point(cir[0], cir[1]), cir[2], cv::Scalar(0,0,255), 2, cv::LINE_AA);
        }

        cv::resize(dst,dst_rz,cv::Size(dst.cols/2,dst.rows/2));

    }

    if(!dst.empty()){
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
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    src1 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/circle2.jpg");
    //    src2 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/f2.jpg");
    //    std::cout << src1.size() << "," << src2.size() << std::endl;
    frame = cv::Mat(680,1024,CV_8UC3);
    roi1 = frame(cv::Rect(setting_width+offset*2,offset,src1.cols / 2,src1.rows / 2));
    //    roi2 = frame(cv::Rect(setting_width+offset*3+src1.cols/2,offset,src2.cols/2,src2.rows/2));

    cv::resize(src1,src1_rz,cv::Size(src1.cols/2,src1.rows/2));


    int dst_x = setting_width + offset*2;
    int dst_y = (offset * 2 + src1.rows / 2);
    dst_roi = frame(cv::Rect(dst_x,dst_y,src1.cols/2,src1.rows/2));
    cvui::init(WINDOW_NAME);
    uv_idle_t idler;
    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);

    uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    cv::destroyAllWindows();
    return 0;

}
