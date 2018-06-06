extern "C" {

    //灰度图像一维数据第一种访问方式
    __global__ void image_add_gray_1(int* img1, int* img2, int* imgres, int length){
        // 一维数据索引计算（万能计算方法）  
        int tid = blockIdx.z * (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) \
                + blockIdx.y * gridDim.x * (blockDim.x * blockDim.y * blockDim.z) \
                + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) \
                + threadIdx.z * (blockDim.x * blockDim.y) \
                + threadIdx.y * blockDim.x \
                + threadIdx.x;  

        if (tid < length) {  
            imgres[tid] = (img1[tid]  + img2[tid]) / 2;  
        }  

    }
    
   //灰度图像一维数据第二种访问方式
   __global__ void image_add_gray_2(int * img1,int * img2,int * imgres,int width,int height){
   
       //Grid中x方向上的索引
       int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        //Grid中y方向上的索引
       int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
       
       int idx = xIndex + yIndex * width;
       
       if (xIndex < width && yIndex < height && idx < width * height){
           imgres[idx] = (img1[idx] + img2[idx]) / 2;
       }
   
   }

   
   //灰度图像带权重方式相加
   __global__ void image_add_gray_weighted(float * img1,float* img2,float * imgres,float alpha,float beta,int width,int height){
         
       //Grid中x方向上的索引
       int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        //Grid中y方向上的索引
       int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
       
       int idx = xIndex + yIndex * width;
       
       if (xIndex < width && yIndex < height){
           imgres[idx] = alpha * img1[idx] + beta * img2[idx];
       }
   }
   
      
   // RGB图像相加
   __global__ void image_add_rgb(int3 * img1,int3 * img2,int3 * imgres,int width,int height){
   
        //Grid中x方向上的索引
       int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        //Grid中y方向上的索引
       int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
       
       int idx = xIndex + yIndex * width;
       
       if (xIndex < width && yIndex < height && idx < width * height){
          int3 rgb1 = img1[idx];
          int3 rgb2 = img2[idx];
          imgres[idx].x = rgb1.x + rgb2.x;
          imgres[idx].y = rgb1.y + rgb2.y;
          imgres[idx].z = rgb1.z + rgb2.z;
       }
   }
   
   
    //RGB图像带权重方式相加
   __global__ void image_add_rgb_weighted(float3 * img1,float3* img2,float3* imgres,float alpha,float beta,int width,int height){
         
       //Grid中x方向上的索引
       int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        //Grid中y方向上的索引
       int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
       
       int idx = xIndex + yIndex * width;
       
       if (xIndex < width && yIndex < height && idx < width * height){
          float3 rgb1 = img1[idx];
          float3 rgb2 = img2[idx];
          imgres[idx].x = alpha * rgb1.x + beta * rgb2.x;
          imgres[idx].y = alpha * rgb1.y + beta * rgb2.y;
          imgres[idx].z = alpha * rgb1.z + beta * rgb2.z;
       }
   }
    //https://blog.csdn.net/hujingshuang/article/details/53115572

}