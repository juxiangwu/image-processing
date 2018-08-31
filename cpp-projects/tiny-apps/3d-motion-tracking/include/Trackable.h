#ifndef TRACKABLE
#define TRACKABLE

namespace Trackable
{
  #define MAX_HISTORY_LENGTH    5

  struct TrackablePoint
  {
    Mat feature;
    deque<Point2f> history;
    int numAbsence; // Number of consecutive frames this point is absent
    int numPresent; // Number of consecutive frames this point is present

    static TrackablePoint create(const Point2f& p, const Mat& feat)
    {
      TrackablePoint t;
      t.feature = feat.clone();
      t.history.push_back(p);
      t.numAbsence = 0;
      t.numPresent = 1;
      return t;
    }

    // TAOTODO: write a velocity function which estimates with 3 consecutive frames

    bool updateNewPosition(const Point2f& p, double momentum, const Mat& newFeat)
    {
      // Update new position by momentum of history
      Point2f np = p;
      int L = history.size();
      if (L>0)
      {
        double sumMomentum = 0;
        for (int i=L-1, k=0; i>=0; i--, k++)
        {
          double c    = pow(momentum, k); // TAOTODO: Add some random noise?
          np          = np + c * history[i];
          sumMomentum += c;
        }
        np /= (1+sumMomentum);
      }
      this->history.push_back(np);
      this->numPresent++;
      this->numAbsence = 0;
      newFeat.copyTo(this->feature);
      if (this->history.size() > MAX_HISTORY_LENGTH)
      {
        this->history.pop_front();
        return false;
      }
      return true;
    }

    int markAbsent()
    {
      this->numAbsence += 1;
      this->numPresent = 0;
      return this->numAbsence;
    }

    Point2f get() const
    {
      return this->history.back();
    }
  };
}

#endif