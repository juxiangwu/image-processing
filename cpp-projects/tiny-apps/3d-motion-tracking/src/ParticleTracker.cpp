#include "ParticleTracker.h"

ParticleTracker::ParticleTracker(string wndName) : ITracker(wndName)
{
  cout << GREEN << "[Tracker]" << RESET << " Initialising SIFT particle tracker..." << endl;

  this->sift      = SIFT::create();
  this->alignment = new Alignment(_dist, this->maxDisplacement);
  this->grid      = nullptr;
  this->alignment->setVisualisation(true);
}

ParticleTracker::~ParticleTracker()
{
  cout << CYAN << "[Tracker]" << RESET << " Terminating SIFT particle tracker..." << endl;
  delete this->sift;
  delete this->alignment;
  delete this->grid;
}

void ParticleTracker::initialiseGrid(int w, int h)
{
  if (this->grid != nullptr)
  {
    cout << YELLOW << "[Tracker] Grid is already initialised. Skipping..." << RESET << endl;
  }
  else
  {
    cout << GREEN << "[Tracker] Initialising Grid" << RESET << endl;
    this->grid = new Grid(Size(w, h), this->maxGravityDistance, this->maxGravityNeighbourhood);
  }
  cout << "[Tracker] Grid initialised " << endl; // TAODEBUG:
}

tuple<vector<Point2f>,Mat> ParticleTracker::detectPoints(Mat &in)
{
  vector<Point2f> points;
  vector<KeyPoint> keypoints;
  Mat descriptors;
  this->sift->detectAndCompute(in, noArray(), keypoints, descriptors);
  KeyPoint::convert(keypoints, points);
  return make_tuple(points, descriptors);
}

function<void (Mat)> ParticleTracker::track()
{
  cout << "[Press Ctrl+c to escape]" << endl;
  auto pipe = [&](Mat im)
  {
    if (this->grid == nullptr)
      this->initialiseGrid(im.cols, im.rows);
    else
      this->grid->neutralise();
    
    trackFeatures(im);
  };

  return pipe;
}

void ParticleTracker::trackFeatures(Mat &im)
{
  auto pointsAndFeatures = detectPoints(im);
  auto points = get<0>(pointsAndFeatures);
  Mat features = get<1>(pointsAndFeatures);
  
  // Draw all detected points as RED
  #ifdef DRAW_ALL_POINTS
  DrawUtils::drawMarks(im, points, Scalar(0,50,255));
  #endif

  unordered_map<int,int> mapNewToOld;
  set<int> trackedPoints;
  set<int> trackedPrevPoints;
  
  if (!this->prevPoints.empty())
  {
    auto pairs = alignment->align(prevPoints, points, features);

    for (auto pair : pairs)
    {
      // Draw connecting line between two aligned points
      int i = pair.first;
      int j = pair.second;
      trackedPoints.insert(j);
      trackedPrevPoints.insert(i);
      mapNewToOld.insert(make_pair(j,i));

      auto pi = prevPoints[i].get();

      if (_dist(pi, points[j]) >= MIN_DISTANCE_TO_DRAW_TRAIL)
        line(im, pi, points[j], Scalar(250,0,0), 1, CV_AA);
      
      #ifndef DRAW_ALL_POINTS
      DrawUtils::drawSpot(im, points[j], Scalar(0,50,255));
      #endif

      // Register the previous points as anchors in the [Grid]
      Point2i p = Point2f(pi.x, pi.y);
      Point2d v = Point2d(points[j].x - pi.x,
                          points[j].y - pi.y);
      this->grid->setAnchor(p, v);
    }

    // Highlight new points with GREEN (without matched previous points)
    #ifdef DRAW_NEW_POINTS_GREEN
    for (int j=0; j<points.size(); j++)
    {
      if (trackedPoints.find(j) == trackedPoints.end())
        DrawUtils::drawSpot(im, points[j], Scalar(0,255,50));
    }
    #endif

    // Given the custom tracking points, generate the induced velocity
    auto velocityMap = this->grid->calculateVelocity(this->trackingPoints);

    // Track the feature points with [Grid]
    this->grid->renderVelocityMap("Grid", velocityMap);

    // Update the new positions of the custom tracking points
    vector<Point2i> newTrackingPoints;
    for (auto vm : velocityMap)
    {
      DrawUtils::drawSpot(im, vm.anchor, Scalar(100,245,0));
      newTrackingPoints.push_back(
        Point2i(vm.anchor.x + vm.velocity.x, 
                vm.anchor.y + vm.velocity.y));
    }
    this->trackingPoints.swap(newTrackingPoints);
  }

  imshow(this->wndName, im);

  // Update the displacement of the positions
  // by momentum
  int numNewPointRegistered = 0;
  int numOldPointAbsent = 0;
  int numTrackedPoints = 0;
  int numDisqualified = 0;
  vector<TrackablePoint> updatedPoints;
  for (int j=0; j<points.size(); j++)
  {
    auto pj = points[j];
    if (mapNewToOld.find(j) == mapNewToOld.end())
    {
      // New point
      updatedPoints.push_back(TrackablePoint::create(pj, features.row(j)));
      numNewPointRegistered++;
    }
    else
    {
      // Tracked point, calculate changes by momentum
      auto pi = prevPoints[mapNewToOld.find(j)->second];
      pi.updateNewPosition(pj, momentum, features.row(j));
      updatedPoints.push_back(pi);
      numTrackedPoints++;
    }
  }

  // Identify old absent points
  for (int i=0; i<prevPoints.size(); i++)
  {
    if (trackedPrevPoints.find(i) == trackedPrevPoints.end())
    {
      // Untracked or missing
      auto pi = prevPoints[i];
      if (pi.markAbsent() <= maxAbsenceAllowed)
      {
        updatedPoints.push_back(pi);
        numOldPointAbsent++;
      }
      else
      {
        numDisqualified++;
      }
    }
  }

  cout << numNewPointRegistered << " new points | "
       << numTrackedPoints << " trackable points | "
       << numOldPointAbsent << " points absent | " 
       << YELLOW << numDisqualified << " points disqualified" << RESET
       << endl;

  // Store the points
  this->prevPoints.swap(updatedPoints);
}