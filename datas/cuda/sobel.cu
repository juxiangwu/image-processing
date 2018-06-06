extern "C"{

__global__ void sobel(float *dataIn, float *dataOut, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * imgWidth + xIndex;
    int Gx = 0;
    int Gy = 0;

    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 
           2 * dataIn[yIndex * imgWidth + xIndex + 1] + 
           dataIn[(yIndex + 1) * imgWidth + xIndex + 1] - 
           (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 
            2 * dataIn[yIndex * imgWidth + xIndex - 1] + 
            dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 
            2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + 
            dataIn[(yIndex - 1) * imgWidth + xIndex + 1] - 
            (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 
            2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + 
            dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}
}