#include "Grid.h"

vector<AnchorWithVelocity> Grid::calculateVelocity(const vector<Point2i>& ps) const
{
  vector<AnchorWithVelocity> psOutput;
  for (auto p : ps)
  {
    // Identify N closest neighbours
    priority_queue<DistanceToAnchor, 
                   vector<DistanceToAnchor>, 
                   CompareNeighbourDistance> closest;
    for (auto c : this->anchors)
    {
      auto distance = _sqrt(_square(c.anchor.x-c.anchor.x) + _square(c.anchor.y-c.anchor.y));
      if (distance > this->maxDistance)
        continue;
      closest.push(make_tuple(distance, c));
    }

    // Take only top N nearest neighbours
    int n = 0;
    double factorRatio = 1.0;
    double totalGravX = 0.0;
    double totalGravY = 0.0;
    double sumFactor = 0.0;
    double closestDistance;
    while (!closest.empty() && n<numNeighbours)
    {
      auto next     = closest.top();
      auto anchor   = get<1>(next);
      auto distance = get<0>(next);
      closest.pop();

      // TAOTOREVIEW: Consider influence from distance to the anchor
      if (n==0) closestDistance = distance;

      double distanceRatio = distance / closestDistance;

      totalGravX += distanceRatio * anchor.velocity.x;
      totalGravY += distanceRatio * anchor.velocity.y;

      n++;
      sumFactor += distanceRatio;
    }

    if (n>0)
    {
      totalGravX /= sumFactor;
      totalGravY /= sumFactor;
    }

    // TAODEBUG:
    cout << " gx : " << totalGravX << endl;
    cout << " gy : " << totalGravY << endl;

    psOutput.push_back(AnchorWithVelocity::create(p, Point2f(totalGravX, totalGravY)));
  }
  return psOutput;
}

void Grid::renderVelocityMap(const string& wndName, const vector<AnchorWithVelocity>& ps)
{
  this->canvas.setTo(Scalar::all(255));

  // Draw anchors first
  for (auto a : this->anchors)
  {
    auto anchor   = a.anchor;
    auto velocity = a.velocity;
    auto vx = velocity.x * PATCH_MAP_SIZE;
    auto vy = velocity.y * PATCH_MAP_SIZE;
    auto aScaled = Point2i(anchor.x*PATCH_MAP_SIZE, anchor.y*PATCH_MAP_SIZE);
    auto bScaled = Point2i((anchor.x + vx)*PATCH_MAP_SIZE, 
                           (anchor.y + vy)*PATCH_MAP_SIZE);
    if (_sqrt(_square(vx) + _square(vy)) > MIN_VELOCITY_TO_DRAW)
      arrowedLine(this->canvas, aScaled, bScaled, Scalar(0,0,245), 1.0, CV_AA, 0, 0.3);

    rectangle(this->canvas, 
              Point2i((anchor.x-2)*PATCH_MAP_SIZE, (anchor.y-2)*PATCH_MAP_SIZE), 
              Point2i((anchor.x+2)*PATCH_MAP_SIZE, (anchor.y+2)*PATCH_MAP_SIZE),
              Scalar(0,0,245),
              1.0, CV_AA);
  }

  // Draw the given points with velocity
  for (auto tup : ps)
  {
    auto p = tup.anchor;
    auto v = tup.velocity;
    auto pScaled = Point2i(p.x*PATCH_MAP_SIZE, p.y*PATCH_MAP_SIZE);
    auto qScaled = Point2i((p.x + v.x)*PATCH_MAP_SIZE, 
                           (p.y + v.y)*PATCH_MAP_SIZE);
    arrowedLine(this->canvas, pScaled, qScaled, Scalar(200,0,0), 1.0, CV_AA);
    rectangle(this->canvas, 
              Point2i((p.x-2)*PATCH_MAP_SIZE, (p.y-2)*PATCH_MAP_SIZE), 
              Point2i((p.x+2)*PATCH_MAP_SIZE, (p.y+2)*PATCH_MAP_SIZE),
              Scalar(0,245,0),
              2.0, CV_AA);
  }

  imshow(wndName, this->canvas);
}