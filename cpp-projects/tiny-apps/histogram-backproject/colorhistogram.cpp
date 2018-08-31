#include "colorhistogram.h"

ColorHistogram::ColorHistogram()
{
    histSize[0]=181;                   //181 values in Hue channel
    hranges[0]=0.0; hranges[1]=180.0;  //max & min of Hue channel
    ranges[0]=hranges;
    channels[0]=0; // use Hue channel only

}

void ColorHistogram::getHueHistogram(const cv::Mat &image,
                                     int minSaturation)
{
    // 转换到HSV空间
    cv::Mat hsv;
    cv::cvtColor(image,hsv,CV_BGR2HSV);

    // 创建掩码矩阵，标出低饱和度像素（S通道像素值<minSaturation）
    cv::Mat mask;
    if(minSaturation>0)
    {
        std::vector<cv::Mat> v;
        cv::split(image,v);
        // 对饱和度（S）通道阈值化
        cv::threshold(v[1],mask,minSaturation,
                255,cv::THRESH_BINARY);
    }

    //1D色调直方图
    cv::calcHist(&hsv,
                 1,
                 channels,
                 mask,  // 掩码矩阵。不计入低饱和度像素
                 hist,
                 1,
                 histSize,
                 ranges);
}
