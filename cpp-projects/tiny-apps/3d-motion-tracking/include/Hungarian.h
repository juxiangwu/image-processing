#ifndef HUNGARIAN
#define HUNGARIAN

#include <vector>
#include <tuple>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.hpp>

using namespace std;
using namespace cv;

// NOTE: The usage of the profile is TRICKY.
//-------------------------------------------
// First element : Index of row, col. (Negative) denotes col
// Secnd element : Number of projected zeroes.
// (All index are 1-based so the sign matters)
typedef tuple<int,int> Profile;

typedef tuple<int,int> ZeroInCol;

class CompareProfile
{
public:
  // TRUE if profile(a) < profile(b)
  bool operator()(Profile &a, Profile &b );
};

class CompareZeroInCol
{
public:
  // TRUE if zeroInCol(a) < zeroInCol(b)
  bool operator()(ZeroInCol &a, ZeroInCol &b);
};

/**
 * Hungarian algorithm implementation 
 * Compatible with [Mat<float>]
 * TAOTOREVIEW: Use template class
 */
class Hungarian
{
public:
  Hungarian(const Mat& cost, bool debug);

  // (Row,Col)
  vector<tuple<int,int>> optimiseMinima() const;

  // Find minimum cover lines to cover all zeroes in matrix {m}
  static tuple<set<int>, set<int>> coverZeroes(Mat& m, bool debug);

  // Create augmented zeroes in the cost matrix {m}
  static void createAdditionalZeros(Mat& m, tuple<set<int>, set<int>> zeroes, bool debug);

private:
  Mat costM;
  bool debug;
  const int MAX_ITER = 8;

  // Find minimum of the i-th row
  static float minOfRow(int i, const Mat& cost);
  // Find minimum of the i-th col
  static float minOfCol(int i, const Mat& cost);
};


#endif