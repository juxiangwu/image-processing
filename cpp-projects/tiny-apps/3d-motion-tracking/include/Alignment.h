#ifndef ALIGNMENT
#define ALIGNMENT

#include <vector>
#include <queue>
#include <tuple>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "master.h"
#include "Distribution.h"
#include "Trackable.h"

using namespace std;
using namespace cv;
using namespace Distribution;
using namespace Trackable;

const int VIS_PATCH_SIZE      = 8;
const int VIS_MAX_SPOT        = 48;
const double FACTOR_SIMILARIY = 0.25;

typedef tuple<double, int> distanceToIndex;

class compareDistance
{
public:
  // Ascending order
  inline bool operator()(distanceToIndex &a, distanceToIndex &b)
  { 
    return get<0>(a) > get<0>(b); 
  }
};

class compareScore
{
public:
  // Descending order
  inline bool operator()(distanceToIndex &a, distanceToIndex &b)
  { 
    return get<0>(a) < get<0>(b); 
  }
};


class Alignment
{
protected:
  bool isVisualisationOn;
  function<double (Point2f, Point2f)> measureDistFunction;
  double maxDistance;
public:
  Alignment(function<double (Point2f, Point2f)> measureDistance, double maxMoveDistance);
  void setVisualisation(bool on);
  unordered_map<int,int> align(vector<TrackablePoint> basepoints, vector<Point2f> newpoints, const Mat newFeatures);
};

#endif