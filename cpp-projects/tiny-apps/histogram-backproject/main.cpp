#include "colorhistogram.h"
#include "contentfinder.h"
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>

void mouseHandler(int event, int x, int y, int flags, void *param);

cv::Rect box = cv::Rect(0,0,-1,-1);
bool drawing_box = false;
bool flag_init = false;
cv::Mat image_sample;

int main()
{
    // 可定制参数
    double th=0.05f;     // 二值化阈值
    int minSaturation=100; // 饱和度阈值
    cv::Mat element1 = cv::Mat(9,9,CV_8U,cv::Scalar(1));
    cv::Mat element2 = cv::Mat(9,9,CV_8U,cv::Scalar(1));
    int minArea = 2000;

    // 实例化两个类
    ColorHistogram hc;
    ContentFinder finder;

    // 摄像头
    cv::VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
    {
        std::cout << "Fail opening the camera!" << std::endl;
        return -1;
    }

    cv::namedWindow("detection");
    cv::namedWindow("Frame");
    cv::setMouseCallback("Frame", mouseHandler);
    cv::Mat frame, image_detect;
    // 等待鼠标圈出目标
    while ((char)cv::waitKey(20) != 'y')
    {
        if ((char)cv::waitKey(20) == 'q')
        {
            cap.release();
            cv::destroyAllWindows();
            return 0;
        }
        cap.read(frame);
        cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);
        cv::imshow("Frame", frame);
    }
    cv::setMouseCallback("Frame", NULL);

    // 计算样本图像直方图
    image_sample = frame(box).clone();
    hc.getHueHistogram(image_sample, minSaturation);

    while ((char)cv::waitKey(20) != 'q')
    {
        cap.read(frame);

        // 识别当前图像低饱和度像素
        std::vector <cv::Mat> v;
        cv::Mat image_hsv;
        cv::cvtColor(frame,image_hsv,CV_BGR2HSV);
        cv::split(image_hsv,v);  //v[1]是饱和度分量
        cv::threshold(v[1],v[1],minSaturation,255,cv::THRESH_BINARY);

        // 寻找目标
        finder.setHistogram(hc.hist);
        finder.setThreshold(th);
        finder.findHue(image_hsv);

        //去除低饱和度像素
        cv::bitwise_and(finder.result,v[1],finder.result);

        // 形态学运算
        cv::dilate(finder.result,finder.result,element1);
        cv::erode(finder.result,finder.result,element2);

        // 直方图反投影初始化矩形框
        if (!flag_init)
        {
            // 提取连通区域轮廓
            image_detect = finder.result.clone();
            std::vector< std::vector<cv::Point> > contours;
            cv::findContours(image_detect,contours,
                             CV_RETR_EXTERNAL,
                             CV_CHAIN_APPROX_NONE);
             // 提取包围盒
             std::vector<cv::Rect> boundRect =
                     std::vector<cv::Rect>(contours.size());
             for(int i=0; i<(int)contours.size(); i++)
             {
                 boundRect[i]=cv::boundingRect(cv::Mat(contours[i]));
             }
             // 选出最大的一个包围盒
             if(contours.size()!=0)
             {
                 for(int i=0,pos_max=0; i<(int)contours.size(); i++)
                 {
                     if(boundRect[pos_max].area() <= boundRect[i].area())
                     {
                         pos_max=i;
                         box=boundRect[pos_max];
                     }
                 }
             }
             flag_init=true;
        }

        // CamShift算法
        if (box.area()>minArea)
        {
            image_detect=finder.result.clone();
            cv::CamShift(image_detect,box,
                cv::TermCriteria(cv::TermCriteria::MAX_ITER,10,0.1));
        }
        else
            flag_init=false;

        // 绘制
        image_detect=frame.clone();
        if (box.area()>minArea)
        {
            cv::rectangle(image_detect,box,cv::Scalar(0,255,0),2);
            cv::circle(image_detect,
                       cv::Point(box.x+box.width/2,box.y+box.height/2),
                       5,cv::Scalar(255,0,0),2);
        }
        else
        {
            cv::rectangle(image_detect,box,cv::Scalar(0,0,255),2);
            cv::circle(image_detect,
                       cv::Point(box.x+box.width/2,box.y+box.height/2),
                       5,cv::Scalar(255,0,0),2);
        }
        cv::imshow("Frame", frame);
        cv::imshow("detection",image_detect);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

void mouseHandler(int event, int x, int y, int flags, void *param)
{
    switch (event)
    {
    case CV_EVENT_MOUSEMOVE:
        if (drawing_box)
        {
            box.width = x - box.x;
            box.height = y - box.y;
        }
        break;
    case CV_EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = cv::Rect(x, y, 0, 0);
        break;
    case CV_EVENT_LBUTTONUP:
        drawing_box = false;
        if (box.width < 0)
        {
            box.x += box.width;
            box.width *= -1;
        }
        if (box.height < 0)
        {
            box.y += box.height;
            box.height *= -1;
        }
        //image_sample = frame(box).clone();
        break;
    }
}
