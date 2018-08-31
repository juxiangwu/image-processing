#ifndef TRACKUTILS
#define TRACKUTILS

#include <vector>
#include <math.h>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cvaux.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class TrackUtils
{
public:
  static vector<Point2f> detectFeaturePoints(Mat& in, int wndSize, int maxCorners, double minDist, double k);
  static Mat detectCorners(Mat& in, int wndSize);


protected:
};

#endif