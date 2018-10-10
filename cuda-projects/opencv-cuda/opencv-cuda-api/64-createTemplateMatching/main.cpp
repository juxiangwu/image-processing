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
#define WINDOW_NAME	"CUDA createTemplateMatching"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame;
cv::Mat src1_rz;
int m_type = 1;
int setting_width = 200,setting_height=300;
int offset = 10;
cv::Mat dst_rz;
void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 200, 300, "Settings");
    src2.copyTo(roi2);
    src1_rz.copyTo(roi1);
    cvui::text(frame, 10, 40, "Mathch Type:CV_TM_SQDIFF");
    cvui::trackbar(frame, 20, 60, 180, &m_type, (int)1, (int)6);
    switch(m_type){
    case 1:
        cvui::text(frame, 10, 40, "Mathch Type:CV_TM_SQDIFF_NORMED");
        break;
    case 2:
        cvui::text(frame, 10, 40, "Mathch Type:CV_TM_SQDIFF_NORMED");
        break;
    case 3:
        cvui::text(frame, 10, 40, "Mathch Type:CV_TM_CCORR");
        break;
    case 4:
        cvui::text(frame, 10, 40, "Mathch Type:CV_TM_CCORR_NORMED");
        break;
    case 5:
        cvui::text(frame, 10, 40, "Mathch Type:CV_TM_CCOEFF");
        break;
    case 6:
        cvui::text(frame, 10, 40, "Mathch Type:CV_TM_CCOEFF_NORMED");
        break;
    }

    if (cvui::button(frame, 20, 230,180,30, "createTemplateMatching")) {
        cv::Ptr<cv::cuda::TemplateMatching> matching;
        switch(m_type){
        case 1:
            matching = cv::cuda::createTemplateMatching(src1.type(),CV_TM_SQDIFF_NORMED);
            break;
        case 2:
            matching = cv::cuda::createTemplateMatching(src1.type(),CV_TM_SQDIFF_NORMED);
            break;
        case 3:
            matching = cv::cuda::createTemplateMatching(src1.type(),CV_TM_CCORR);
            break;
        case 4:
            matching = cv::cuda::createTemplateMatching(src1.type(),CV_TM_CCORR_NORMED);
            break;
        case 5:
            matching = cv::cuda::createTemplateMatching(src1.type(),CV_TM_CCOEFF);
            break;
        case 6:
            matching = cv::cuda::createTemplateMatching(src1.type(),CV_TM_CCOEFF_NORMED);
            break;
        }
        cv::cuda::GpuMat dev_src(src1),dev_template(src2),dev_res;
        matching->match(dev_src,dev_template,dev_res);
        cv::Mat result;
        dev_res.download(result);
        cv::Point minLoc,maxLoc;
        double min_val,max_val;
        cv::minMaxLoc(result,&min_val,&max_val,&minLoc,&maxLoc);
        cv::Rect rect;
        if(m_type == CV_TM_SQDIFF || m_type == CV_TM_SQDIFF_NORMED){
            rect = cv::Rect(minLoc, src2.size());
        }else{
            rect = cv::Rect(maxLoc, src2.size());
        }

//        cv::Mat result;
        src1.copyTo(dst);
        cv::rectangle(dst,rect,cv::Scalar(0,0,255),2);
//        cv::imshow("dst",dst);
        cv::resize(dst,dst_rz,cv::Size(dst.cols / 2,dst.rows/2));
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
    src1 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/plane.jpg");
    src2 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/plane_part_2.jpg");
//    std::cout << src1.size() << "," << src2.size() << std::endl;
    frame = cv::Mat(680,1024,CV_8UC3);
    roi1 = frame(cv::Rect(setting_width+offset*2,offset,src1.cols / 2,src1.rows / 2));
    roi2 = frame(cv::Rect(setting_width+offset*3+src1.cols/2,offset,src2.cols,src2.rows));

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
