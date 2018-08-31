#ifndef CONTENTFINDER_H
#define CONTENTFINDER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ContentFinder
{
public:
    ContentFinder();
    void setThreshold(double t);
    void setHistogram(const cv::MatND &h);
    void findHue(const cv::Mat &image);

public:
    cv::Mat result;

private:
    float hranges[2] ;
    const float* ranges[1] ;
    int channels[1] ;
    float threshold ;
    cv::MatND histogram ;
};

#endif // CONTENTFINDER_H
