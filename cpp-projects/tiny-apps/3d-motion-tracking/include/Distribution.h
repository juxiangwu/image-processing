#ifndef DISTRIBUTION
#define DISTRIBUTION

#include <assert.h>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "master.h"

using namespace std;
using namespace cv;

namespace Distribution
{
  /**
   * Bucket defines an interval of continuous space of type <T>
   */
  template<class T> class Bucket
  {
  protected:
    T lowerBound;
    T upperBound;
    T interval; // Resolution of the bucket along the line space
  public:
    Bucket(T interval, T lowerBound, T upperBound);

    // Index of the bucket which the specified value belongs to
    virtual int index(const T v);
    virtual int numBucketsWithinInterval();
  };

  template<class T> class GenericDistribution 
  {
  private:
    vector<T> population;
  public:
    inline void clear()
    {
      this->population.clear();
    };

    inline void addPopulation(T e)
    {
      this->population.push_back(e);
    };

    virtual void bucketPlot(Bucket<T> bucketInterval, tuple<T,T> bounds, string wnd, int drawUnitSize, T cutOffLine);
    
  };

}

#endif