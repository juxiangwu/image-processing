#ifndef MESH_OBJECT
#define MESH_OBJECT

#include <queue>
#include <vector>
#include <math.h>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cvaux.hpp>
#include "Particle.h"
#include "Hungarian.h"
#include "DrawUtils.h"
#include "CalcUtils.h"

// DEPRECATED:

struct Point2fPair
{
  Point2f a, b;
  double distance;
};

struct ShorterDistance
{
  bool operator() (const Point2fPair& p1, const Point2fPair& p2) const 
  {
    return p1.distance < p2.distance;
  }
};

class MeshObject
{
public:
  MeshObject(){};
  MeshObject(vector<Point2f> &ps);

  Point2f centroid() const; 
  Particle centroidAsParticle() const;

  // Split the mesh into one or more sub-meshes
  vector<MeshObject> split(double maxEdgeLength=100) const;

  bool isEmpty() const;
  int lengthOfAbsence;
  inline int lengthOfHistory() const { return this->history.size(); }

  void drawMesh(Mat &canvas, Scalar edgeColor, Scalar vertexColor, double maxDistance=100, bool thick=false) const;
  void drawHistoryPath(Mat &canvas, Scalar pathColor) const;
  void update(const MeshObject &newM);
  inline const vector<Point2f> toVec() const { return this->points; };

protected:
  vector<Point2f> points; // TAOTOREVIEW: KD-Tree might be a better choice
  vector<MeshObject> history; // Temporal history of the mesh

  // Find minimum distance from a point [p]
  // to the group of points [ps]
  static double findMinDistance(Point2f p, vector<Point2f> ps);
};




#endif 