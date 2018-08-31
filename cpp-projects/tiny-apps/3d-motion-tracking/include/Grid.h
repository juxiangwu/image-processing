#ifndef GRID
#define GRID

#include <math.h>
#include <vector>
#include <tuple>
#include <functional>
#include <opencv2/opencv.hpp>
#include "master.h"

using namespace std;
using namespace cv;

struct AnchorWithVelocity
{
  Point2d   anchor;
  Point2d   velocity;

  static AnchorWithVelocity create(Point2d anc, Point2d v)
  {
    AnchorWithVelocity a;
    a.anchor   = anc;
    a.velocity = v;
    return a;
  }
};
typedef tuple<double, AnchorWithVelocity> DistanceToAnchor;


class CompareNeighbourDistance
{
public:
  // Ascending order
  inline bool operator()(DistanceToAnchor &a, DistanceToAnchor &b)
  { 
    return get<0>(a) > get<0>(b); 
  }
};

class Grid
{
private:
  const int PATCH_MAP_SIZE = 2;

protected:
  Size size;
  vector<AnchorWithVelocity> anchors;
  double maxDistance; // Maximum distance from the anchor which gravity can take effect
  unsigned int numNeighbours; // Number of closest neighbours to take into account

  Mat canvas;

public:
  Grid(Size gridSize, double maxInfluentialDistance, unsigned int neighbours)
  {
    this->size          = gridSize;
    this->maxDistance   = maxInfluentialDistance;
    this->numNeighbours = neighbours;
    this->canvas        = Mat(gridSize.height * PATCH_MAP_SIZE, gridSize.width * PATCH_MAP_SIZE, CV_8UC3);
  }
  
  virtual ~Grid()
  {
  }

  inline void neutralise()
  {
    this->anchors.clear();
  }

  inline void setAnchor(const Point2f& anchor, const Point2d& velocity)
  {
    this->anchors.push_back(AnchorWithVelocity::create(anchor, velocity));
  }

  // Calculate velocity with direction on the specified list of subjects
  virtual vector<AnchorWithVelocity> calculateVelocity(const vector<Point2i>& ps) const;

  virtual void renderVelocityMap(const string& wndName, const vector<AnchorWithVelocity>& ps);
};



#endif