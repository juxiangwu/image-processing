#include "contentfinder.h"

ContentFinder::ContentFinder()
{
    hranges[0]=0.0; hranges[1]=180.0;
    ranges[0]=hranges;
    channels[0]=0;
    threshold=-1.0f;
}

void ContentFinder::setThreshold(double t)
{
    threshold=t;
}

void ContentFinder::setHistogram(const cv::MatND &hist)
{
    histogram=hist;
    cv::normalize(histogram,histogram,1.0);
}

void ContentFinder::findHue(const cv::Mat &image)
{
    cv::calcBackProject(&image,
                        1,
                        channels,
                        histogram,
                        result,
                        ranges,
                        255.0);

    if((threshold>0)&&(threshold<1))
    {
        cv::threshold(result,result,
                      255*threshold,255,
                      cv::THRESH_BINARY);
    }
}
