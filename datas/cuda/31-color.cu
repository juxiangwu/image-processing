extern "C" {

__global__ void rgb2yiq(float3 * src,float3 * dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = 0.5950059 * rgb.x - 0.27455667 * rgb.y - 0.32134392 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 0.21153661 * rgb.x - 0.52273617 * rgb.y + 0.31119955 * rgb.z;
    }
}

__global__ void yiq2rgb(float3 * src,float3 * dst,int imgWidth,int imgHeight){

    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 1.00000001 * rgb.x + 0.95598634 * rgb.y + 0.6208248 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = 0.9999999 * rgb.x - 0.27201283 * rgb.y - 0.64720424 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 1.00000002 * rgb.x - 1.10674021 * rgb.y + 1.70423049 * rgb.z;
    }
}

__global__ void rgb2yuv(float3*src,float3 *dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = -0.14714119 * rgb.x - 0.28886916 * rgb.y + 0.43601035 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 0.61497538 * rgb.x - 0.51496512 * rgb.y - 0.10001026 * rgb.z;
    }
}

__global__ void yuv2rgb(float3*src,float3 *dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex].x = 1.0 * rgb.x + 0.0 * rgb.y + 1.13988303 * rgb.z;
        dst[yIndex * imgWidth + xIndex].y = 1.0 * rgb.x - 0.395 * rgb.y - 0.58062185 * rgb.z;
        dst[yIndex * imgWidth + xIndex].z = 1.0 * rgb.x + 2.03206185 * rgb.y - 0.00000000 * rgb.z;
    }
}

__global__ void rgb2gray_avg(float3*src,float*dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex] = (rgb.x + rgb.y + rgb.z)/3.0;
    }
    

}

__global__ void rgb2gray(float3*src,float*dst,int imgWidth,int imgHeight){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        dst[yIndex * imgWidth + xIndex] = 0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z;
    }
    

}



__global__ void rgb2YCbCr(float3 * src,float3 * dst,int imgWidth,int imgHeight){

    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    
    const float YCbCrYRF = 0.299;      // RGB转YCbCr的系数(浮点类型）
    const float YCbCrYGF = 0.587;
    const float YCbCrYBF = 0.114;
    const float YCbCrCbRF = -0.168736;
    const float YCbCrCbGF = -0.331264;
    const float YCbCrCbBF = 0.500000;
    const float YCbCrCrRF = 0.500000;
    const float YCbCrCrGF = -0.418688;
    const float YCbCrCrBF = -0.081312;

    const float RGBRYF = 1.00000 ;   // YCbCr转RGB的系数(浮点类型）
    const float RGBRCbF = 0.0000;
    const float RGBRCrF = 1.40200;
    const float RGBGYF = 1.00000  ;
    const float RGBGCbF = -0.34414;
    const float RGBGCrF = -0.71414;
    const float RGBBYF = 1.00000  ;
    const float RGBBCbF = 1.77200;
    const float RGBBCrF = 0.00000 ;

    const int Shift = 20;
    const int HalfShiftValue = 1 << (Shift - 1);

    const int YCbCrYRI = (int)((YCbCrYRF * (1 << Shift) + 0.5)); // RGB转YCbCr的系数(整数类型）
    const int YCbCrYGI = (int)((YCbCrYGF * (1 << Shift) + 0.5));
    const int YCbCrYBI = (int)((YCbCrYBF * (1 << Shift) + 0.5));
    const int YCbCrCbRI = (int)((YCbCrCbRF * (1 << Shift) + 0.5));
    const int YCbCrCbGI = (int)((YCbCrCbGF * (1 << Shift) + 0.5));
    const int YCbCrCbBI = (int)((YCbCrCbBF * (1 << Shift) + 0.5));
    const int YCbCrCrRI = (int)((YCbCrCrRF * (1 << Shift) + 0.5));
    const int YCbCrCrGI = (int)((YCbCrCrGF * (1 << Shift) + 0.5));
    const int YCbCrCrBI = (int)((YCbCrCrBF * (1 << Shift) + 0.5));
/*
    const int RGBRYI = (int)((RGBRYF * (1 << Shift) + 0.5)) ;     // YCbCr转RGB的系数(整数类型）
    const int RGBRCbI = (int)((RGBRCbF * (1 << Shift) + 0.5));
    const int RGBRCrI = (int)((RGBRCrF * (1 << Shift) + 0.5));
    const int RGBGYI = (int)((RGBGYF * (1 << Shift) + 0.5));
    const int RGBGCbI = (int)((RGBGCbF * (1 << Shift) + 0.5));
    const int RGBGCrI = (int)((RGBGCrF * (1 << Shift) + 0.5));
    const int RGBBYI = (int)((RGBBYF * (1 << Shift) + 0.5));
    const int RGBBCbI = (int)((RGBBCbF * (1 << Shift) + 0.5));
    const int RGBBCrI = (int)((RGBBCrF * (1 << Shift) + 0.5)) ;
*/    
    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        int Red = rgb.x;
        int Green = rgb.y;
        int Blue = rgb.z;
        float Y = (float)((YCbCrYRI * Red + YCbCrYGI * Green + YCbCrYBI * Blue + HalfShiftValue) >> Shift);
        float Cb = (float)( 128 + ( (YCbCrCbRI * Red + YCbCrCbGI * Green + YCbCrCbBI * Blue + HalfShiftValue) >> Shift));
        float Cr = (float)(128+( (YCbCrCrRI * Red + YCbCrCrGI * Green + YCbCrCrBI * Blue + HalfShiftValue) >> Shift));
        
       dst[yIndex * imgWidth + xIndex].x = Y;
       dst[yIndex * imgWidth + xIndex].y = Cb;
       dst[yIndex * imgWidth + xIndex].z = Cr;
       
    }
}

__global__ void YCrCb2RGB(float3*src,float3*dst,int imgWidth,int imgHeight){
int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    
    const float YCbCrYRF = 0.299;      // RGB转YCbCr的系数(浮点类型）
    const float YCbCrYGF = 0.587;
    const float YCbCrYBF = 0.114;
    const float YCbCrCbRF = -0.168736;
    const float YCbCrCbGF = -0.331264;
    const float YCbCrCbBF = 0.500000;
    const float YCbCrCrRF = 0.500000;
    const float YCbCrCrGF = -0.418688;
    const float YCbCrCrBF = -0.081312;

    const float RGBRYF = 1.00000 ;   // YCbCr转RGB的系数(浮点类型）
    const float RGBRCbF = 0.0000;
    const float RGBRCrF = 1.40200;
    const float RGBGYF = 1.00000  ;
    const float RGBGCbF = -0.34414;
    const float RGBGCrF = -0.71414;
    const float RGBBYF = 1.00000  ;
    const float RGBBCbF = 1.77200;
    const float RGBBCrF = 0.00000 ;

    const int Shift = 20;
    const int HalfShiftValue = 1 << (Shift - 1);

    const int YCbCrYRI = (int)((YCbCrYRF * (1 << Shift) + 0.5)); // RGB转YCbCr的系数(整数类型）
    const int YCbCrYGI = (int)((YCbCrYGF * (1 << Shift) + 0.5));
    const int YCbCrYBI = (int)((YCbCrYBF * (1 << Shift) + 0.5));
    const int YCbCrCbRI = (int)((YCbCrCbRF * (1 << Shift) + 0.5));
    const int YCbCrCbGI = (int)((YCbCrCbGF * (1 << Shift) + 0.5));
    const int YCbCrCbBI = (int)((YCbCrCbBF * (1 << Shift) + 0.5));
    const int YCbCrCrRI = (int)((YCbCrCrRF * (1 << Shift) + 0.5));
    const int YCbCrCrGI = (int)((YCbCrCrGF * (1 << Shift) + 0.5));
    const int YCbCrCrBI = (int)((YCbCrCrBF * (1 << Shift) + 0.5));

    const int RGBRYI = (int)((RGBRYF * (1 << Shift) + 0.5)) ;     // YCbCr转RGB的系数(整数类型）
    const int RGBRCbI = (int)((RGBRCbF * (1 << Shift) + 0.5));
    const int RGBRCrI = (int)((RGBRCrF * (1 << Shift) + 0.5));
    const int RGBGYI = (int)((RGBGYF * (1 << Shift) + 0.5));
    const int RGBGCbI = (int)((RGBGCbF * (1 << Shift) + 0.5));
    const int RGBGCrI = (int)((RGBGCrF * (1 << Shift) + 0.5));
    const int RGBBYI = (int)((RGBBYF * (1 << Shift) + 0.5));
    const int RGBBCbI = (int)((RGBBCbF * (1 << Shift) + 0.5));
    const int RGBBCrI = (int)((RGBBCrF * (1 << Shift) + 0.5)) ;
 
    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        float3 rgb = src[yIndex * imgWidth + xIndex];
        
        int Y = rgb.x;
        int Cb = rgb.y - 128;
        int Cr = rgb.z - 128;
        float Red = Y + ((RGBRCrI * Cr + HalfShiftValue) >> Shift);
        float Green = Y + ((RGBGCbI * Cb + RGBGCrI * Cr+ HalfShiftValue) >> Shift);
        float Blue = Y + ((RGBBCbI * Cb + HalfShiftValue) >> Shift);
       dst[yIndex * imgWidth + xIndex].x = Red;
       dst[yIndex * imgWidth + xIndex].y = Green;
       dst[yIndex * imgWidth + xIndex].z = Blue;
       
    }
}

}