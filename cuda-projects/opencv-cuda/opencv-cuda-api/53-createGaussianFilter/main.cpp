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
#define WINDOW_NAME	"CUDA createGaussianFilter"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame;
int ksize = 1;
double sigma = 0.0;
void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 200, 300, "Settings");
    cv::Mat src1_rz,src2_rz;
    cv::resize(src1,src1_rz,cv::Size(src1.cols/2,src1.rows/2));
    cv::resize(src2,src2_rz,cv::Size(src2.cols/2,src2.rows/2));
    src1_rz.copyTo(roi1);

    cvui::text(frame, 10, 40, "ksize");
    cvui::trackbar(frame, 20, 60, 180, &ksize, (int)1, (int)15);
    cvui::text(frame,10,140,"sigma");
    cvui::trackbar(frame, 20, 160, 180, &sigma, (double)0., (double)5.0);
    if (cvui::button(frame, 20, 230,180,30, "createGaussianFilter")) {
//        cv::Mat gray;
        std::vector<cv::Mat>channels;
        std::vector<cv::Mat>dst_channels;
        cv::split(src1,channels);
        for(int i = 0;i < channels.size();i++){
            cv::Mat gray = channels[i];
//            gray.convertTo(gray,CV_32F);
            int kz = ksize * 2 + 1;
            cv::Ptr<cv::cuda::Filter> filter =
                    cv::cuda::createGaussianFilter(gray.type(),gray.type(),
                                              cv::Size(kz,kz),sigma,0);
            cv::Mat res;
            cv::cuda::GpuMat dev_gray(gray);
            cv::cuda::GpuMat dev_res;
            filter->apply(dev_gray,dev_res);
            dev_res.download(res);
            dst_channels.push_back(res);
        }
        cv::merge(dst_channels,dst);
        dst.convertTo(dst,CV_8UC3);
        cv::convertScaleAbs(dst,dst);
//        dst.copyTo(dst_roi);
//        cv::imshow("dst",res);
    }



    if(!dst.empty()){
        cv::Mat dst_rz;
        cv::resize(dst,dst_rz,cv::Size(dst.cols / 2,dst.rows/2));
        dst_rz.copyTo(roi2);
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
    src1 = cv::imread("../../../../datas/f1.jpg");
    src2 = cv::imread("../../../../datas/f2.jpg");
    frame = cv::Mat(600,1024,CV_8UC3);
    roi1 = frame(cv::Rect(230,10,src1.cols / 2,src1.rows / 2));
    std::cout<<"roi1:" << cv::Rect(100,10,src1.cols / 2,src1.rows / 2) << std::endl;
    roi2 = frame(cv::Rect(240+src1.cols / 2,10,src1.cols / 2,src1.rows / 2));
    std::cout<<"roi2:" << cv::Rect(110+src1.cols / 2,10,src1.cols / 2,src1.rows / 2) << std::endl;
    int dst_x = (src1.cols+10) / 2;
    int dst_y = (20 + src1.rows / 2);
    dst_roi = frame(cv::Rect(dst_x,dst_y,src1.cols/2,src1.rows/2));
    cvui::init(WINDOW_NAME);
    uv_idle_t idler;
    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);

    uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    cv::destroyAllWindows();
    return 0;

}
