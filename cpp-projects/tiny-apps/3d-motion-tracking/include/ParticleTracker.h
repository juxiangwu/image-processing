#ifndef PARTICLE_TRACKER
#define PARTICLE_TRACKER

#include <set>
#include <tuple>
#include <vector>
#include <deque>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "master.h"
#include "ITracker.h"
#include "DrawUtils.h"
#include "Alignment.h"
#include "Grid.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


class ParticleTracker : public ITracker
{
private:
  const int maxAbsenceAllowed                = 3;
  const double maxDisplacement               = 12;
  const double maxGravityDistance            = 500;
  const unsigned int maxGravityNeighbourhood = 8;
  const double momentum                      = 0.25;

  Alignment* alignment;
  vector<TrackablePoint> prevPoints;
  Grid* grid;

protected:
  Ptr<Feature2D> sift;
  void trackFeatures(Mat &im);
  void initialiseGrid(int w, int h);

public:
  ParticleTracker(string wndName);
  ~ParticleTracker();
  tuple<vector<Point2f>,Mat> detectPoints(Mat &in);

  function<void (Mat)> track();
};



#endif