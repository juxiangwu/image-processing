extern "C" {

__global__ void distoring_mirror(float3 * src,float3 * dst,int imgWidth,int imgHeight,int x,int y){
    
    float radius = 0.0;
    float theta = 0.0;
    float map_x = 0.0;
    float map_y = 0.0;
    float map_r = 0.0;
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = yIndex * imgWidth + xIndex;
    if (xIndex < imgWidth && yIndex < imgHeight)
    {
       int tx = xIndex - x;
       int ty = yIndex - y;
       theta = atan2f(ty,tx);
       radius = sqrtf(tx * tx + ty * ty);
       map_r = sqrtf(radius * 100);
       map_x = (int)(x + map_r * cosf(theta));
       map_y = (int)(y + map_r * sinf(theta));
       
       int new_offset = map_y * imgWidth + map_x;
       dst[offset].x = src[new_offset].x;
       dst[offset].y = src[new_offset].y;
       dst[offset].z = src[new_offset].z;
    }
}

}