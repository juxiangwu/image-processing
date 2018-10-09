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
cv::Ptr<cv::cudacodec::VideoReader> videoReader;
void wait_for_a_while(uv_idle_t* handle){

    bool success = videoReader->nextFrame(video_frame);
    std::cout << "read frame:" << success << std::endl;
    if(success){
        cv::resize(video_frame,video_frame,
                   cv::Size(video_frame.cols/2,video_frame.rows / 2));
        video_frame.copyTo(roi1);
    }else{
//        uv_idle_stop(handle);
//        exit(0);
        std::cout << "cannot read frame\n";
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
    videoReader = cv::cudacodec::createVideoReader(cv::String("d:/develop/dl/projects/resources/videos/motor_bike.mp4"));
    frame = cv::Mat(600,1024,CV_8UC3);
    roi1 = frame(cv::Rect(180,10,1280 / 2,720 / 2));
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
