extern "C" { 
    
__global__ void gamma_transform_2(float3* d_idata, float3* d_odata, int width, int height, float gamma)  
    {  
      int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
      int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
      int idx = yIndex * width + xIndex;
     if (xIndex < width && yIndex < height){
        float3 rgb = d_idata[idx];
        d_odata[idx].x = powf(rgb.x, gamma);
        d_odata[idx].y = powf(rgb.y, gamma);
        d_odata[idx].z = powf(rgb.z, gamma);
     }
           
    }

}