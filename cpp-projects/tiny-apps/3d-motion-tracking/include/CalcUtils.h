#ifndef CALC_UTILS
#define CALC_UTILS

#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

inline double _square(double n)
{
  return n*n;
}

// Fast square root
// [href] : http://www.codeproject.com/Articles/69941/Best-Square-Root-Method-Algorithm-Function-Precisi
inline double _sqrt(double n)
{
  union
  {
    int i;
    double x;
  } u;
  u.x = n;
  u.i = (1<<29) + (u.i >> 1) - (1<<22); 

  // Two Babylonian Steps (simplified from:)
  // u.x = 0.5f * (u.x + n/u.x);
  // u.x = 0.5f * (u.x + n/u.x);
  u.x =       u.x + n/u.x;
  u.x = 0.25f*u.x + n/u.x;

  return u.x;
}


inline double _dist(Point2f p1, Point2f p2)
{
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return _sqrt(dx*dx + dy*dy);
}



#endif