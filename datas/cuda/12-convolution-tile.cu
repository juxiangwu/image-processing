extern "C" {


    #define TILE_SIZE 12
    #define NUMBER_THREAD_X 16
    #define NUMBER_THREAD_Y 16
    #define TILE_SIZE NUMBER_THREAD_X * NUMBER_THREAD_Y * 3 // each block matches with the input tile
    __global__ void convolution_tile(float *I, float *P,float * deviceMaskData,int width, int height,int channels, int maskRows,int maskColumns) {
        const int maskRowsRadius = maskRows / 2;
        const int maskColumnsRadius = maskColumns / 2;
        // Original columns/rows index before shifting
        int colOriginal = blockIdx.x * (blockDim.x - maskColumnsRadius*2) + threadIdx.x;
        int rowOriginal = blockIdx.y * (blockDim.y - maskRowsRadius*2) + threadIdx.y;
        
        // Thread columns and rows
        // (Original cols/rows shifted by the mask radius backwards)
        int colT = colOriginal - maskColumnsRadius;
        int rowT = rowOriginal - maskRowsRadius;

        int depth = threadIdx.z;

        // 1st phase: copy from global memory to shared memory (tiling)

        // As design choice, we assume that each block matches each input tile
        // meaning that each thread loads its own input pixel
        // but only the central ones computes the output pixel
        __shared__ float Ids[TILE_SIZE];
        int sharedMemoryPos = (threadIdx.y * blockDim.y + threadIdx.x)*channels + depth;

        // Actual tiling
        if (rowT >= 0 && rowT < height && colT >= 0 && colT < width) {
            Ids[sharedMemoryPos] = I[(rowT * width + colT) * channels + depth];
        }
        else { // check for ghost elements
            Ids[sharedMemoryPos] = 0.0f;
        }

        // Wait for other threads in the same block
        __syncthreads();

        // 2nd phase: evaluate convolution

        // This first IF is to check whether we're still inside the image boundaries or not
        if (rowT >= 0 && rowT < height && colT >= 0 && colT < width) {
            // This second IF is to check whether we're inside the central block area or not (border threads do not compute anything)
            if (threadIdx.x >= maskColumnsRadius && threadIdx.x < (blockDim.x - 2) && threadIdx.y >= maskRowsRadius && threadIdx.y < (blockDim.y - 2)) {
                float pValue = 0;

                int startCol = threadIdx.x - maskColumnsRadius;
                int startRow = threadIdx.y - maskRowsRadius;

                for (int i = 0; i < maskRows; i++) {
                    for (int j = 0; j < maskColumns; j++) {
                        int currentCol = startCol + j;
                        int currentRow = startRow + i;

                        // Check for ghost elements already done during tiling
                        float iValue = Ids[(currentRow * blockDim.y + currentCol) * channels + depth];

                        pValue += iValue * deviceMaskData[i * maskRows + j];
                    }
                }

                // Store the result inside the output vector P in the global memory
                P[(rowT * width + colT) * channels + depth] = pValue;
            }
        }
    }
}