extern "C" {
__global__ void img_reverse(uchar3* d_idata, uchar3* d_odata, int width, int height){

      int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
      int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
      int idx = yIndex * width + xIndex;
     if (xIndex < width && yIndex < height){
        uchar3 rgb = d_idata[idx];
        d_odata[idx].x = 255 - rgb.x;
        d_odata[idx].y = 255 - rgb.y;
        d_odata[idx].z = 255 - rgb.z;
     }
}

}