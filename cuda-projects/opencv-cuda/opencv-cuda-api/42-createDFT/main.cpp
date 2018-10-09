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
#define WINDOW_NAME	"CUDA createDFT"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame;


void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 120, 200, "Settings");
    cv::Mat src1_rz,src2_rz;
    cv::resize(src1,src1_rz,cv::Size(src1.cols/2,src1.rows/2));
    cv::resize(src2,src2_rz,cv::Size(src2.cols/2,src2.rows/2));
    src1_rz.copyTo(roi1);
    //    src2_rz.copyTo(roi2);

    if (cvui::button(frame, 15, 50,100,30, "createDFT")) {
//        cv::Ptr<cv::cuda::DFT> dft = cv::cuda::createDFT(cv::Size(3,3),cv::DFT_ROWS);
//        //        cv::Mat kernel_x = (cv::Mat_<float>(3,3) << 2, 0, -2, 4, 0, -4, 2, 0, -2);
        cv::Mat res;
        cv::Mat gray;
        cv::cvtColor(src1,gray,cv::COLOR_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
//        //        conv->convolve(gray,kernel_x,res);
//        dft->compute(gray,res);
//        //        std::cout << res.size() << std::endl;
//        ////        cv::convertScaleAbs(gray,gray);
//        cv::cvtColor(res,res,cv::COLOR_GRAY2BGR);
//        cv::convertScaleAbs(res,res);
//        dst = cv::Mat::zeros(res.size(),res.type());
//        //        res.copyTo(dst);
//        cv::imshow("dst",dst);
        cv::cuda::dft(gray,res,cv::Size(3,3));


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
    src2 = cv::imread("../../../../datas/f2.jpg");
    frame = cv::Mat(600,1024,CV_8UC3);
    roi1 = frame(cv::Rect(180,10,src1.cols / 2,src1.rows / 2));
    std::cout<<"roi1:" << cv::Rect(100,10,src1.cols / 2,src1.rows / 2) << std::endl;
    roi2 = frame(cv::Rect(190+src1.cols / 2,10,src1.cols / 2,src1.rows / 2));
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
