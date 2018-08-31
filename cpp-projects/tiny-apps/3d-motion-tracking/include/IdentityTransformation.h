#include <opencv2/opencv.hpp>
#include <opencv/cvaux.hpp>
#include "ITransformation.h"

#pragma once

using namespace std;
using namespace cv;

/**
 * A transformation which does not do anything
 * with the input. Just simply passes it through.
 */
class IdentityTransformation : public ITransformation
{
public:
  IdentityTransformation();
  ~IdentityTransformation();

  Mat transform(Mat& input);
};