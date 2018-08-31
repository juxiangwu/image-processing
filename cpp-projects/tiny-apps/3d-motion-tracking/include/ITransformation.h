#include <opencv2/opencv.hpp>
#include <opencv/cvaux.hpp>

#pragma once

using namespace cv;

/**
 * Abstract matrix or image transformation interface
 * 
 * {Mat} -> [Transformation T] -> {Mat}
 */
class ITransformation
{
public:
  virtual Mat transform(Mat& input) = 0;
};