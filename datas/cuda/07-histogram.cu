extern "C" {

    //灰度直方图统计
    __global__ void histogram(unsigned char *dataIn, int *hist)
    {
        int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
        int blockIndex = blockIdx.x + blockIdx.y * gridDim.x;
        int index = threadIndex + blockIndex * blockDim.x * blockDim.y;
        atomicAdd(&hist[dataIn[index]], 1);

    }
}