extern "C" {

    //灰度直方图统计
    __global__ void histogram(unsigned char *dataIn, int *hist)
    {
        int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
        int blockIndex = blockIdx.x + blockIdx.y * gridDim.x;
        int index = threadIndex + blockIndex * blockDim.x * blockDim.y;
        atomicAdd(&hist[dataIn[index]], 1);

    }
    
   //灰度图像直方图（优化）
    __global__ void histogram_optimized(unsigned char *buffer, long size, unsigned int *histo){
        __shared__ unsigned int private_histo[256];
        if(threadIdx.x < 256)  //初始化shared histo
            private_histo[threadIdx.x] = 0;
        __syncthreads();

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        // 步长是所有threads的数目
        int stride = blockDim.x * gridDim.x;
        while(i < size) {
            atomicAdd(&(private_histo[buffer[i]]), 1);
            i += stride;
        }

        //等待所有线程执行完
        __syncthreads();

        if(threadIdx.x < 256){
            atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);
        }
    }
}