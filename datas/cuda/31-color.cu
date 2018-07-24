extern "C" {

__global__ void rgb2yiq(float3 * src,float3 * dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = 0.596 * rgb.x - 0.274 * rgb.y - 0.322 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 0.211 * rgb.x - 0.523 * rgb.y + 0.312 * rgb.z;
    }
}

__global__ void yiq2rgb(float3 * src,float3 * dst,int imgWidth,int imgHeight){

    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 1.0 * rgb.x + 0.956 * rgb.y + 0.621 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = 1.0 * rgb.x - 0.272 * rgb.y - 0.647 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 1.0 * rgb.x - 1.106 * rgb.y - 1.703 * rgb.z;
    }
}

__global__ void rgb2yuv(float3*src,float3 *dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = -0.148 * rgb.x - 0.289 * rgb.y + 0.437 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 0.615 * rgb.x - 0.515 * rgb.y - 0.100 * rgb.z;
    }
}

__global__ void yuv2rgb(float3*src,float3 *dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 1.0 * rgb.x + 0.0 * rgb.y + 1.140 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = 1.0 * rgb.x - 0.395 * rgb.y - 0.581 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 1.0 * rgb.x + 2.032 * rgb.y - 0.000 * rgb.z;
    }
}

}