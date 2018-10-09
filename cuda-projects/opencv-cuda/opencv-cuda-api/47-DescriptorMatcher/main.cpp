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
#include "opencv2/xfeatures2d/cuda.hpp"

#define CVUI_IMPLEMENTATION
#include <cvui.h>
extern "C"{
#include <uv.h>
}

using namespace std;
#define WINDOW_NAME	"CUDA reduce"
cv::Mat src1,src2,dst,roi1,roi2,dst_roi,frame,gray1,gray2;
cv::cuda::SURF_CUDA surf;

void wait_for_a_while(uv_idle_t* handle){
    frame = cv::Scalar(49, 52, 49);
    cvui::window(frame, 10, 10, 120, 200, "Settings");
    cv::Mat src1_rz,src2_rz;
    cv::resize(src1,src1_rz,cv::Size(src1.cols/2,src1.rows/2));
    cv::resize(src2,src2_rz,cv::Size(src2.cols/2,src2.rows/2));
    src1_rz.copyTo(roi1);
    //    src2_rz.copyTo(roi2);

    if (cvui::button(frame, 15, 50,100,30, "process")) {

        cv::cuda::GpuMat img1(gray1), img2(gray2);

        cv::cuda::GpuMat keypoints1GPU, keypoints2GPU;
        cv::cuda::GpuMat descriptors1GPU, descriptors2GPU;
        surf(img1, cv::cuda::GpuMat(), keypoints1GPU, descriptors1GPU);
        surf(img2, cv::cuda::GpuMat(), keypoints2GPU, descriptors2GPU);

        cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
                cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
        std::vector<cv::DMatch> matches;
        matcher->match(descriptors1GPU, descriptors2GPU, matches);

        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        std::vector<float> descriptors1, descriptors2;
        surf.downloadKeypoints(keypoints1GPU, keypoints1);
        surf.downloadKeypoints(keypoints2GPU, keypoints2);
        surf.downloadDescriptors(descriptors1GPU, descriptors1);
        surf.downloadDescriptors(descriptors2GPU, descriptors2);

        cv::Mat img_matches;
        cv::drawMatches(cv::Mat(img1), keypoints1,
                        cv::Mat(img2), keypoints2, matches, img_matches);

        cv::resize(img_matches,img_matches,cv::Size(img_matches.cols/2,img_matches.rows/2));
        dst_roi = frame(cv::Rect(10,src1.rows/2+10,img_matches.cols/2,
                                 img_matches.rows/2));

        cv::imshow("dst",img_matches);
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
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    src1 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/plane.jpg");
    src2 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/plane_part.jpg");
    gray1 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/plane.jpg",0);
    gray2 = cv::imread("D:/Develop/DL/projects/digital-image-processing/datas/plane_part.jpg",0);
    frame = cv::Mat(600,1024,CV_8UC3);
    roi1 = frame(cv::Rect(180,10,src1.cols / 2,src1.rows / 2));
    std::cout<<"roi1:" << cv::Rect(100,10,src1.cols / 2,src1.rows / 2) << std::endl;
    roi2 = frame(cv::Rect(190+src1.cols / 2,10,src1.cols / 2,src1.rows / 2));
    cvui::init(WINDOW_NAME);
    uv_idle_t idler;
    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);

    uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    cv::destroyAllWindows();
    return 0;

}
