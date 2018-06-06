extern "C"{

    __global__ void threshold(unsigned char * src,unsigned char * dst,int width,int height,int thresh){
          //Grid中x方向上的索引
          int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
          //Grid中y方向上的索引
          int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

          int idx = xIndex + yIndex * width;

          if (xIndex < width && yIndex < height && idx < width * height){
              if (src[idx] > thresh){
                  dst[idx] = 255;
              }else{
                 dst[idx] = 0;
              }
          }

    }
    
      __global__ void multi_threshold(unsigned char * src,unsigned char * dst,int width,int height,int min_thresh,int max_thresh){
          //Grid中x方向上的索引
          int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
          //Grid中y方向上的索引
          int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

          int idx = xIndex + yIndex * width;

          if (xIndex < width && yIndex < height && idx < width * height){
              int pixel = src[idx];
              if (pixel >= min_thresh &&  pixel <= max_thresh){
                  dst[idx] = 255;
              }else{
                 dst[idx] = 0;
              }
          }

    }

}