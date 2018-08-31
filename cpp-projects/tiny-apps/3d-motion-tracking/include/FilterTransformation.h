#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include "ITransformation.h"

#pragma once

using namespace std;
using namespace cv;

enum FilterType
{
  Nothing,
  Gray,
  BinaryThreshold,
  TruncateThreshold,
  ZeroThreshold
};

class FilterTransformation : public ITransformation
{
public:
  FilterTransformation(FilterType t, float level, float maxScale, float ratioSize);

  Mat transform(Mat& input);

protected:
  FilterTransformation();
  FilterType m_type;
  float m_level;
  float m_maxScale;
  float m_ratioSize;

  virtual Mat toGray(Mat &input) final;
  virtual Mat toBinarised(Mat &input) final;
  virtual Mat toTruncated(Mat &input) final;
  virtual Mat toZeroCut(Mat &input) final;
  Mat reshape(Mat &input);

};