#include <iostream>
#include <opencv2/opencv.hpp>

#define CVUI_IMPLEMENTATION
#include <cvui.h>
extern "C"{
#include <uv.h>
}

using namespace std;
#define WINDOW_NAME	"libuv camera"
cv::Mat win_frame,cap_frame,cap_roi;
cv::VideoCapture cap(0);

void wait_for_a_while(uv_idle_t* handle){
    win_frame = cv::Scalar(49, 52, 49);
    cap >> cap_frame;
    cv::resize(cap_frame,cap_frame,cv::Size(640,480));
    if(cap_frame.empty()){
        std::cout << "cannot read frame from camera.\n";
    }else{
        cap_frame.copyTo(cap_roi);
    }

    cvui::update();

    // Show everything on the screen
    cv::imshow(WINDOW_NAME, win_frame);

    // Check if ESC was pressed
    if (cv::waitKey(30) == 27) {
        uv_idle_stop(handle);
    }
}

int main()
{
    if(!cap.isOpened()){
        std::cerr << "cannot open camera.\n";
        return 0;
    }
    win_frame = cv::Mat(600,1024,CV_8UC3);
    cap_roi = win_frame(cv::Rect(180,10,640,480));
    cvui::init(WINDOW_NAME);
    uv_idle_t idler;
    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);

    uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    cv::destroyAllWindows();
    return 0;

}
