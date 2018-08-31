#ifndef ITRACKER
#define ITRACKER

#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>

using namespace cv;
using namespace std;

class ITracker 
{
protected:
  // Custom tracking points
  vector<Point2i> trackingPoints;

public:
  string wndName;
  ITracker(string wndName)
  {
    this->wndName = wndName;
    namedWindow(wndName);
  }
  virtual inline ~ITracker(){};
  virtual function<void (Mat)> track() = 0;
  virtual void addTrackingPoint(const Point2i& p)
  {
    this->trackingPoints.push_back(p);
    cout << this->trackingPoints.size() << " points being tracked" << endl; // TAODEBUG:
  }

  static void bindMouseEvent(string wndName, MouseCallback ev)
  {
    setMouseCallback(wndName, ev, nullptr);
  }
};

#endif