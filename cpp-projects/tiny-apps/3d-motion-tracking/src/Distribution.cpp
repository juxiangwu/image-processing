#include "Distribution.h"

template<class T>
Distribution::Bucket<T>::Bucket(T interval, T lowerBound, T upperBound)
{
  assert(lowerBound < upperBound);
  this->interval   = interval;
  this->lowerBound = lowerBound;
  this->upperBound = upperBound;
}

template<class T>
int Distribution::Bucket<T>::index(const T v)
{
  return (int)floor((v - this->lowerBound)/this->interval);
}

template<class T>
int Distribution::Bucket<T>::numBucketsWithinInterval()
{
  return (this->upperBound - this->lowerBound)/this->interval;
}

template<class T>
void Distribution::GenericDistribution<T>::bucketPlot(Bucket<T> bucketInterval, tuple<T,T> bounds, string wnd, int drawUnitSize, T cutOffLine)
{
  int bucketCount = bucketInterval.numBucketsWithinInterval();
  Mat buckets = Mat::zeros(1, bucketCount, CV_32FC1);

  // Collect bucket population ticks
  for (auto p : this->population)
  {
    int index = bucketInterval.index(p);
    if (index < bucketCount)
    {
      float v = buckets.at<float>(0, index);
      buckets.at<float>(0, index) = v+1.0;
    } 
  }
  
  normalize(buckets, buckets, 0.0, 1.0, NORM_MINMAX);


  int M = 500;
  int N = drawUnitSize * bucketCount;
  Mat plot = Mat(M, N, CV_8UC3, Scalar(0,0,0));

  auto prevX = 0;
  auto prevY = M - (int)floor(M * buckets.at<float>(0, 0));
  for (int b=0; b<bucketCount; b++)
  {
    auto yPos = M - (int)floor(M * buckets.at<float>(0, b));
    auto xPos = (drawUnitSize * b) + (int)floor(drawUnitSize/2);
    line(
      plot, 
      Point2f(prevX, prevY), 
      Point2f(xPos, yPos), 
      Scalar(0,50,200), 
      1, CV_AA);
    circle(plot, Point2f(xPos,yPos), 3, Scalar(0,30,180), CV_FILLED, CV_AA);
    prevX = xPos;
    prevY = yPos;
  }
  // Draw cut off line if in boundary
  int indexCutOff = bucketInterval.index(cutOffLine);
  if (indexCutOff>=0 && indexCutOff<bucketCount)
  {
    auto x  = (drawUnitSize * indexCutOff) + (int)floor(drawUnitSize/2);
    line(
      plot,
      Point2f(x,0),
      Point2f(x,M),
      Scalar(255,255,255),
      1, CV_AA);
  }
  imshow(wnd, plot);
}

template class Distribution::Bucket<double>;
template class Distribution::GenericDistribution<double>;