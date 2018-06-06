extern "C"{

//腐蚀
__global__ void erode(float *dataIn, float *dataOut, int erodeElementWidth,int erodeElementHeight, int imgWidth, int imgHeight)
{
    //Grid中x方向上的索引
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    //Grid中y方向上的索引
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    int elementWidth = erodeElementWidth;
    int elementHeight = erodeElementHeight;
    int halfEW = elementWidth / 2;
    int halfEH = elementHeight / 2;
    //初始化输出图
    dataOut[yIndex * imgWidth + xIndex] = dataIn[yIndex * imgWidth + xIndex];
    //防止越界
    if (xIndex > halfEW && xIndex < imgWidth - halfEW && yIndex > halfEH && yIndex < imgHeight - halfEH)
    {
        for (int i = -halfEH; i < halfEH + 1; i++)
        {
            for (int j = -halfEW; j < halfEW + 1; j++)
            {
                if (dataIn[(i + yIndex) * imgWidth + xIndex + j] >= dataOut[yIndex * imgWidth + xIndex])
                {
                    dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
                }
            }
        }
    }
}

//腐蚀
__global__ void dilate(float *dataIn, float *dataOut, int erodeElementWidth,int erodeElementHeight, int imgWidth, int imgHeight)
{
    //Grid中x方向上的索引
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    //Grid中y方向上的索引
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    int elementWidth = erodeElementWidth;
    int elementHeight = erodeElementHeight;
    int halfEW = elementWidth / 2;
    int halfEH = elementHeight / 2;
    //初始化输出图
    dataOut[yIndex * imgWidth + xIndex] = dataIn[yIndex * imgWidth + xIndex];;
    //防止越界
    if (xIndex > halfEW && xIndex < imgWidth - halfEW && yIndex > halfEH && yIndex < imgHeight - halfEH)
    {
        for (int i = -halfEH; i < halfEH + 1; i++)
        {
            for (int j = -halfEW; j < halfEW + 1; j++)
            {
                if (dataIn[(i + yIndex) * imgWidth + xIndex + j] < dataOut[yIndex * imgWidth + xIndex])
                {
                    dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
                }
            }
        }
    }
}

}