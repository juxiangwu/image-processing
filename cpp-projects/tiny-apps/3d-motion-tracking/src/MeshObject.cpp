#include "MeshObject.h"

MeshObject::MeshObject(vector<Point2f> &ps)
{
  this->points = ps;
  this->lengthOfAbsence = 0;
}

Particle MeshObject::centroidAsParticle() const
{
  Point2f centr = this->centroid();
  return Particle(centr);
}

Point2f MeshObject::centroid() const
{
  if (this->isEmpty())
    throw "No elements";
  double sumx = 0;
  double sumy = 0;
  double N    = (double)this->points.size();

  for (Point2f p : this->points)
  {
    sumx += p.x;
    sumy += p.y;
  }

  return Point2f(sumx/N, sumy/N);
}

double MeshObject::findMinDistance(Point2f p, vector<Point2f> ps)
{
  double minDist = numeric_limits<double>::max();
  for (Point2f q : ps)
  {
    double d = _dist(p, q);
    minDist = d < minDist ? d : minDist;
  }
  return minDist;
}

vector<MeshObject> MeshObject::split(double maxEdgeLength) const
{
  vector<MeshObject> ms;

  // Traverse through the mesh graph 
  vector<Point2f> points0 = this->toVec();
  vector<Point2f> residues;

  while (!points0.empty())
  {
    vector<Point2f> neighbors;

    // Pick up next candidate
    Point2f p0 = points0.back();
    points0.pop_back();
    neighbors.push_back(p0);

    // Traverse through the remaining points
    while (!points0.empty())
    {
      Point2f p = points0.back();
      points0.pop_back();
      double d  = MeshObject::findMinDistance(p, neighbors);

      // If the point is close to the current aggregating mesh, 
      // count it in
      if (d<maxEdgeLength)
        neighbors.push_back(p);
      else
        residues.push_back(p);
    }

    // Done
    MeshObject m(neighbors);
    ms.push_back(m);

    // [residues]  becomes next [points0]
    points0.clear();
    while (!residues.empty()) 
    {
      Point2f p = residues.back();
      residues.pop_back();
      points0.push_back(p);
    }
  }

  return ms;
}

bool MeshObject::isEmpty() const
{
  return this->points.empty();
}


void MeshObject::drawMesh(Mat &canvas, Scalar edgeColor, Scalar vertexColor, double maxDistance, bool thick) const
{
  auto N = this->points.size();
  // Draw edges
  for (int i=0; i<N; i++)
    for (int j=i+1; j<N; j++)
    {
      Point2f p0 = this->points[i];
      Point2f p  = this->points[j];
      if (_dist(p0,p) <= maxDistance)
        line(canvas, p0, p, edgeColor, 1, CV_AA);
    }

  double bx0 = numeric_limits<double>::max();
  double bxN = -numeric_limits<double>::max();
  double by0 = numeric_limits<double>::max();
  double byN = -numeric_limits<double>::max();

  // Draw vertices
  for (Point2f p : this->points)
  {
    circle(canvas, p, 1, vertexColor, thick ? 3 : 1, CV_AA);
    // Update bounding box coordinates
    bx0 = p.x < bx0 ? p.x : bx0;
    by0 = p.y < by0 ? p.y : by0;
    bxN = p.x > bxN ? p.x : bxN;
    byN = p.y > byN ? p.y : byN;
  }

  // Draw bounding box
  rectangle(canvas, Point2f(bx0, by0), Point2f(bxN, byN), vertexColor, thick ? 3 : 1);
}

void MeshObject::update(const MeshObject &newM)
{
  // Current state becomes the most recent history
  this->history.push_back(MeshObject(this->points));
  this->points.clear();

  const vector<Point2f> ps = newM.toVec();

  // New mesh becomes the current state
  copy(ps.begin(), ps.end(), back_inserter(this->points));
}

void MeshObject::drawHistoryPath(Mat &canvas, Scalar pathColor) const
{
  if (this->history.size()>0)
  {
    arrowedLine(
      canvas,
      this->history.back().centroid(),
      this->centroid(),
      pathColor,
      1.0,
      CV_AA
    );
  }
}


