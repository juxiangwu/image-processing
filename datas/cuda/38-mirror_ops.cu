extern "C" {
    /* 
    * Kernel for mirroring the image parallely usng CUDA
    */
    __global__ 
    void mirror(const uchar4* const inputChannel, uchar4* outputChannel, int numRows, int numCols, bool vertical)
    {
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      int row = blockIdx.y * blockDim.y + threadIdx.y;

      if ( col >= numCols || row >= numRows )
      {
       return;
      }

      if(!vertical)
      { 

        int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
        int thread_y = blockDim.y * blockIdx.y + threadIdx.y;

        int thread_x_new = thread_x;
        int thread_y_new = numRows-thread_y;

        long myId = thread_y * numCols + thread_x;
        long myId_new = thread_y_new * numCols + thread_x_new;
        outputChannel[myId_new] = inputChannel[myId];

      }

      else
      {
          unsigned int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
          unsigned int thread_y = blockDim.y * blockIdx.y + threadIdx.y;

          unsigned int thread_x_new = numCols-thread_x;
          unsigned int thread_y_new = thread_y;

          unsigned long int myId = thread_y * numCols + thread_x;
          unsigned long int myId_new = thread_y_new * numCols + thread_x_new;
        //printf("Id : %lu\t NewId : %lu\n", myId, myId_new);

        outputChannel[myId_new] = inputChannel[myId];  // linear data store in global memory	
      }
}  

}