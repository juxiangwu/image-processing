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
#define WINDOW_NAME	"CUDA demosaicing"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame;
cv::Mat src1_rz,src2_rz;
int m_type = 1;
int setting_width = 200,setting_height=300;
int offset = 10;
cv::Mat dst_rz;
void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 200, 300, "Settings");
    src2_rz.copyTo(roi2);
    src1_rz.copyTo(roi1);
    cvui::text(frame, 10, 40, "COLOR_BayerBG2BGR_MHT ");
    cvui::trackbar(frame, 20, 60, 180, &m_type, (int)1, (int)12);
    switch(m_type){
    case 1:
        cvui::text(frame, 10, 40, "COLOR_BayerBG2BGR_MHT ");
        break;
    case 2:
        cvui::text(frame, 10, 40, "COLOR_BayerGB2BGR_MHT");
        break;
    case 3:
        cvui::text(frame, 10, 40, "COLOR_BayerRG2BGR_MHT");
        break;
    case 4:
        cvui::text(frame, 10, 40, "COLOR_BayerGR2BGR_MHT");
        break;
    case 5:
        cvui::text(frame, 10, 40, "COLOR_BayerBG2RGB_MHT");
        break;
    case 6:
        cvui::text(frame, 10, 40, "COLOR_BayerGB2RGB_MHT");
        break;
    case 7:
        cvui::text(frame, 10, 40, "COLOR_BayerRG2RGB_MHT");
        break;
    case 8:
        cvui::text(frame, 10, 40, "COLOR_BayerGR2RGB_MHT");
        break;
    case 9:
        cvui::text(frame, 10, 40, "COLOR_BayerBG2GRAY_MHT");
        break;
    case 10:
        cvui::text(frame, 10, 40, "COLOR_BayerGB2GRAY_MHT");
        break;
    case 11:
        cvui::text(frame, 10, 40, "COLOR_BayerRG2GRAY_MHT");
        break;
    case 12:
        cvui::text(frame, 10, 40, "COLOR_BayerGR2GRAY_MHT");
        break;

    }

    if (cvui::button(frame, 20, 230,180,30, "demosaicing")) {

        std::vector<cv::Mat> channels;
        std::vector<cv::Mat> dst_channels;
        cv::split(src1,channels);
        for(int i = 0; i<channels.size();i++){
            cv::cuda::GpuMat dev_src1(channels[i]),dev_res;

            switch(m_type){
            case 1:
                cvui::text(frame, 10, 40, "COLOR_BayerBG2BGR_MHT ");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerBG2BGR_MHT);
                break;
            case 2:
                cvui::text(frame, 10, 40, "COLOR_BayerGB2BGR_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerGB2BGR_MHT);
                break;
            case 3:
                cvui::text(frame, 10, 40, "COLOR_BayerRG2BGR_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerRG2BGR_MHT);
                break;
            case 4:
                cvui::text(frame, 10, 40, "COLOR_BayerGR2BGR_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerGR2BGR_MHT);
                break;
            case 5:
                cvui::text(frame, 10, 40, "COLOR_BayerBG2RGB_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerBG2RGB_MHT);
                break;
            case 6:
                cvui::text(frame, 10, 40, "COLOR_BayerGB2RGB_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerGB2RGB_MHT);
                break;
            case 7:
                cvui::text(frame, 10, 40, "COLOR_BayerRG2RGB_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerRG2RGB_MHT);
                break;
            case 8:
                cvui::text(frame, 10, 40, "COLOR_BayerGR2RGB_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerGR2RGB_MHT);
                break;
            case 9:
                cvui::text(frame, 10, 40, "COLOR_BayerBG2GRAY_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerBG2GRAY_MHT);
                break;
            case 10:
                cvui::text(frame, 10, 40, "COLOR_BayerGB2GRAY_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerGB2GRAY_MHT);
                break;
            case 11:
                cvui::text(frame, 10, 40, "COLOR_BayerRG2GRAY_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerRG2GRAY_MHT);
                break;
            case 12:
                cvui::text(frame, 10, 40, "COLOR_BayerGR2GRAY_MHT");
                cv::cuda::demosaicing(dev_src1,dev_res,cv::cuda::COLOR_BayerGR2GRAY_MHT);
                break;

            }
            cv::Mat res;
            dev_res.download(res);
            dst_channels.push_back(res);
        }
        //        cv::cvtColor(dst,dst,cv::COLOR_BGRA2BGR);
        std::cout << "channel.size="<<dst_channels.size()<<std::endl;
        cv::merge(dst_channels,dst);
        std::cout << dst.channels() << std::endl;
//        cv::imshow("dst",dst);
//        cv::resize(dst,dst_rz,cv::Size(dst.cols / 2,dst.rows/2));

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
    src1 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/f1.jpg");
    src2 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/f2.jpg");
    //    std::cout << src1.size() << "," << src2.size() << std::endl;
    frame = cv::Mat(680,1024,CV_8UC3);
    roi1 = frame(cv::Rect(setting_width+offset*2,offset,src1.cols / 2,src1.rows / 2));
    roi2 = frame(cv::Rect(setting_width+offset*3+src1.cols/2,offset,src2.cols/2,src2.rows/2));

    cv::resize(src1,src1_rz,cv::Size(src1.cols/2,src1.rows/2));
    cv::resize(src2,src2_rz,cv::Size(src2.cols/2,src2.rows/2));

    int dst_x = setting_width + offset*2;
    int dst_y = (offset * 2 + src1.rows / 2);
    dst_roi = frame(cv::Rect(dst_x,dst_y,src2.cols/2,src2.rows/2));
    cvui::init(WINDOW_NAME);
    uv_idle_t idler;
    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);

    uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    cv::destroyAllWindows();
    return 0;

}
