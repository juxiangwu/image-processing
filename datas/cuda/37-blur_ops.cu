extern "C" {
    /*
    * Kernel for blurring the image parallely usng CUDA- general
    */
    __global__
    void gaussian_blur_1(const uchar3* const inputChannel, uchar3* outputChannel,
        int numRows, int numCols, const float* const filter, const int filterWidth)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (col >= numCols || row >= numRows)
        {
            return;
        }

        long myId = row * numCols + col;
        float result_x = 0.f;
        float result_y = 0.f;
        float result_z = 0.f;

        for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; filter_r++)
        {

            for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; filter_c++)
            {

                float image_value_x = static_cast<float>(inputChannel[myId].x);
                float filter_value = filter[(filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2];
                result_x += image_value_x * filter_value;

                float image_value_y = static_cast<float>(inputChannel[myId].y);
                result_y += image_value_y * filter_value;

                float image_value_z = static_cast<float>(inputChannel[myId].z);
                result_z += image_value_z * filter_value;

            }
        }
        uchar4 pix = make_uchar3(result_x, result_y, inputChannel[myId].z);
        outputChannel[row * numCols + col] = pix;
    }
    
        /*
    * Kernel for blurring the image parallely usng CUDA- RGB in parallel
    */
    __global__
    void gaussian_blur(const unsigned char* const inputChannel,
        unsigned char* const outputChannel,
        int numRows, int numCols,
        const float* const filter, const int filterWidth)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (col >= numCols || row >= numRows) {
            return;
        }

        float c = 0.0f;

        for (int fx = 0; fx < filterWidth; fx++) {
            for (int fy = 0; fy < filterWidth; fy++) {
                int imagex = col + fx - filterWidth / 2;
                int imagey = row + fy - filterWidth / 2;
                imagex = min(max(imagex, 0), numCols - 1);
                imagey = min(max(imagey, 0), numRows - 1);
                c += (filter[fy*filterWidth + fx] * inputChannel[imagey*numCols + imagex]);
            }
        }

        outputChannel[row*numCols + col] = c;
    }

}