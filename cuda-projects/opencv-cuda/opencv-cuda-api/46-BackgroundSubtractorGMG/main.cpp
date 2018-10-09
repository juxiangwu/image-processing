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
#define WINDOW_NAME	"CUDA BackgroundSubtractorFGD"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame,video_frame;
cv::Ptr<cv::cuda::BackgroundSubtractorGMG> gmg;
cv::VideoCapture cap;
void wait_for_a_while(uv_idle_t* handle){

    cap >> video_frame;

    cv::resize(video_frame,video_frame,
               cv::Size(video_frame.cols/2,video_frame.rows/2));

    video_frame.copyTo(roi1);
    cv::cuda::GpuMat device_frame(video_frame);
    cv::cuda::GpuMat device_dst;
    gmg->apply(device_frame,device_dst);
    device_dst.download(dst);
    //    std::cout << dst.size() << std::endl;
    if(!dst.empty()){
        cv::cvtColor(dst,dst,cv::COLOR_GRAY2BGR);
        dst.copyTo(roi2);
    }
    //    cv::imshow("dst",dst);
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
    cap = cv::VideoCapture("d:/develop/dl/projects/resources/videos/768x576.avi");
    if(!cap.isOpened()){
        std::cout << "cannot open video file\n";
        return -1;
    }
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    gmg = cv::cuda::createBackgroundSubtractorGMG(40);
    frame = cv::Mat(600,1024,CV_8UC3);
    roi1 = frame(cv::Rect(180,10,768 / 2,576 / 2));
    roi2 = frame(cv::Rect(190+768 / 2,10, 768 / 2,576 / 2));
    int dst_x = (768+10) / 2;
    int dst_y = (20 + 576 / 2);
    dst_roi = frame(cv::Rect(dst_x,dst_y,768/2,576/2));
    cvui::init(WINDOW_NAME);
    uv_idle_t idler;
    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);
    uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    cv::destroyAllWindows();
    return 0;

}
