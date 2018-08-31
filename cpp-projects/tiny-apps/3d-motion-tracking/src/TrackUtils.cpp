#include "TrackUtils.h"

Mat TrackUtils::detectCorners(Mat& in, int wndSize)
{
  Mat h = Mat::zeros(in.size(), CV_32FC1);
  int blockSize    = 9;
  int apertureSize = wndSize;
  double k         = 0.20;
  cornerHarris(in, h, blockSize, apertureSize, k);
  return h;
}

vector<Point2f> TrackUtils::detectFeaturePoints(Mat& in, int wndSize, int maxCorners, double minDist, double k)
{
  vector<Point2f> corners;
  double quality   = 0.015;
  Mat mask         = Mat();
  
  // Shi-Tomasi
  goodFeaturesToTrack(in, corners, maxCorners, quality, minDist, mask, wndSize, false, k);
  return corners;
}
