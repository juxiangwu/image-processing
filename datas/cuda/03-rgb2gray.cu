extern "C" {

__global__ void rgb2gray(uchar3 *dataIn, unsigned char *dataOut, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        uchar3 rgb = dataIn[yIndex * imgWidth + xIndex];

        dataOut[yIndex * imgWidth + xIndex] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

}