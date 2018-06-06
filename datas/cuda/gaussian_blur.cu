extern "C" {

__global__
void gaussian_blur(const  float* const inputChannel,
                    float* const outputChannel,
                   int numRows, int numCols, const float* const filter, const int filterWidth)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if ( col >= numCols || row >= numRows )
  {
   return;
  }

  float result = 0.f;
    //For every value in the filter around the pixel (c, r)
    for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) 
    {
      for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) 
      {
        //Find the global image position for this filter position
        //clamp to boundary of the image
        int image_r = min(max(row + filter_r, 0), static_cast<int>(numRows - 1));
        int image_c = min(max(col + filter_c, 0), static_cast<int>(numCols - 1));

        float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
        float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

        result += image_value * filter_value;
      }
    }
  outputChannel[row * numCols + col] = result;
}

}