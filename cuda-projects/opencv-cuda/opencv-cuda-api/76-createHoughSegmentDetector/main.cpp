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
#define WINDOW_NAME	"CUDA HoughSegmentDetector"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame;
cv::Mat src1_rz;
int m_type = 1;
int setting_width = 200,setting_height=300;
int offset = 10;
cv::Mat dst_rz;
cv::Ptr<cv::cuda::HoughSegmentDetector> hough;

void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 200, 300, "Settings");
    //    src2.copyTo(roi2);
    src1_rz.copyTo(roi1);

    if (cvui::button(frame, 20, 230,180,30, "HoughSegmentDetector")) {
        cv::Mat gray;
        cv::cvtColor(src1,gray,cv::COLOR_BGR2GRAY);
        cv::cuda::GpuMat dev_src(gray),dev_dst,dev_dst_lines;
        cv::cuda::threshold(dev_src,dev_dst,180,240,cv::THRESH_BINARY);
        vector<cv::Vec4i> lines;

        hough = cv::cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 50, 5);

        hough->detect(dev_dst,dev_dst_lines);
        std::cout << dev_dst_lines.cols << std::endl;
        if (!dev_dst_lines.empty())
        {
            lines.resize(dev_dst_lines.cols);
            cv::Mat h_lines(1, dev_dst_lines.cols, CV_32SC4, &lines[0]);
            dev_dst_lines.download(h_lines);
        }
        src1.copyTo(dst);
        for( size_t i = 0; i < lines.size(); i++ )
        {
            cv::Vec4i line_point  = lines[i];
            cv::line(dst, cv::Point(line_point[0], line_point[1]),
                    cv::Point(line_point[2], line_point[3]),
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
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
    src1 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/angles.jpg");
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
