#include "IdentityTransformation.h"

IdentityTransformation::IdentityTransformation()
{
}

IdentityTransformation::~IdentityTransformation()
{
}

Mat IdentityTransformation::transform(Mat &input)
{
  return input;
}