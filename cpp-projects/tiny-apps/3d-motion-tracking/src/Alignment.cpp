#include "Alignment.h"
#define _USE_MATH_DEFINES
#include <math.h>
Alignment::Alignment(function<double (Point2f, Point2f)> measureDistance, double maxMoveDistance)
{
  cout << GREEN << "Initialising alignment engine..." << RESET << endl;
  this->measureDistFunction  = measureDistance;
  this->maxDistance          = maxMoveDistance;
  this->isVisualisationOn    = false;
}

void Alignment::setVisualisation(bool on)
{
  this->isVisualisationOn = on;
}

unordered_map<int,int> Alignment::align(vector<TrackablePoint> basepoints, vector<Point2f> newpoints, const Mat newFeatures)
{
  int i = 0;
  int M = VIS_MAX_SPOT * VIS_PATCH_SIZE + 1;
  Mat matchScore = Mat::zeros(basepoints.size(), newpoints.size(), CV_64FC1);
  Mat vis = Mat::zeros(M, M, CV_8UC3);
  unordered_map<int,int> pairs;
  
  // Record the population distribution (of scores)
  GenericDistribution<double> scorePopulation;
  for (auto bp0 : basepoints)
  {
    auto bp = bp0.get();

    // List of Tuples of <distance, index of candidate>
    // Closest first
    priority_queue<distanceToIndex, vector<distanceToIndex>, compareScore> candidates;

    // Measure all candidates within the perimeter
    int j = 0;
    for (auto np : newpoints)
    {
      double d = this->measureDistFunction(bp, np);
      if (d > this->maxDistance)
      {
        // NOTE: Do not add zero as a candidate
        matchScore.at<double>(i,j) = 0;
      }
      else
      {
        auto v0 = bp0.feature;
        auto v1 = newFeatures.row(j);
        double mag0 = norm(v0, CV_L2);
        double mag1 = norm(v1, CV_L2);
        double similarity = FACTOR_SIMILARIY * 0.5 * (M_PI - acos(v0.dot(v1)/(mag0*mag1)))/M_PI;

        double score = (d<=1) ? 1.0 : similarity / pow(d,2.0);
        if (score > 1e-20)
          candidates.push(make_tuple(j, score));
        matchScore.at<double>(i,j) = score;

        // Record the score population
        scorePopulation.addPopulation(score);
        
        if (this->isVisualisationOn)
        {
          int v   = floor(255 * score);
          auto p0 = Point2f(i*VIS_PATCH_SIZE, j*VIS_PATCH_SIZE);
          auto p1 = Point2f((i+1)*VIS_PATCH_SIZE-1, (j+1)*VIS_PATCH_SIZE-1);
          auto c  = Scalar(0,0,int(v));
          rectangle(vis, p0, p1, c, -1);
        }
      }
      j++;
    }

    if (!candidates.empty())
    {
      auto matchedPoint = candidates.top();
      if (get<0>(matchedPoint) > 0)
        pairs.insert(make_pair(i, get<0>(matchedPoint)));
    }

    i++;
  }

  // Reject candidates of matching with low scores here
  Mat meanVec, stdVec;
  meanStdDev(matchScore, meanVec, stdVec);
  double mean = meanVec.at<double>(0,0);
  double std  = stdVec.at<double>(0,0);
  unordered_map<int,int> pairsFiltered;
  const double rejectedStdTimes = 4;
  int numRejected = 0;
  for (auto pair : pairs)
  {
    auto score = pair.second;
    if (score > mean + rejectedStdTimes*std)
    {
      pairsFiltered.insert(pair);
    }
    else numRejected++;
  }

  #ifdef DEBUG_ALIGNMENT
  cout << numRejected << " candidates rejected by mean : " << mean
    << " (+" << rejectedStdTimes*std << ")" << endl;
  #endif
  
  if (this->isVisualisationOn)
  {
    imshow("matching score", vis);
    auto binstep = Bucket<double>(0.01, 0.0, 1.0);
    auto bounds  = make_tuple(0.0, 0.1);
    #ifdef DEBUG_ALIGNMENT
    scorePopulation.bucketPlot(binstep, bounds, "Score distribution", 3, mean + rejectedStdTimes*std);
    #endif
  }

  return pairsFiltered;
}
