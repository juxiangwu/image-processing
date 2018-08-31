#include "FilterTransformation.h"

FilterTransformation::FilterTransformation(FilterType t, float level, float maxScale, float ratioSize)
{
  this->m_type     = t;
  this->m_level    = level;
  this->m_maxScale = maxScale;
  this->m_ratioSize = ratioSize;
}

Mat FilterTransformation::transform(Mat& input)
{
  switch (m_type)
  {
    case FilterType::Gray: 
      return toGray(input);
    case FilterType::BinaryThreshold: 
      return toBinarised(input);
    case FilterType::TruncateThreshold:
      return toTruncated(input);
    case FilterType::ZeroThreshold:
      return toZeroCut(input);
    default:
      return reshape(input); // By default, analogous to IdentityTransformation
  }
}

Mat FilterTransformation::reshape(Mat &input)
{
  if (this->m_ratioSize == 1) return input;
  Mat buffer;
  resize(input, buffer, Size(input.cols*m_ratioSize, input.rows*m_ratioSize));
  return buffer;  
}

Mat FilterTransformation::toGray(Mat &input)
{
  Mat buffer;
  cvtColor(input, buffer, CV_BGR2GRAY);
  return reshape(buffer);
}

Mat FilterTransformation::toBinarised(Mat &input)
{
  Mat buffer;
  threshold(input, buffer, m_level, m_maxScale, THRESH_BINARY);
  return reshape(buffer);
}

Mat FilterTransformation::toTruncated(Mat &input)
{
  Mat buffer;
  threshold(input, buffer, m_level, m_maxScale, THRESH_TRUNC);
  return reshape(buffer);
}

Mat FilterTransformation::toZeroCut(Mat &input)
{
  Mat buffer;
  threshold(input, buffer, m_level, m_maxScale, THRESH_TOZERO);
  return reshape(buffer);
}