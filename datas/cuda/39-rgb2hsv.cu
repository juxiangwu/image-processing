extern "C" {

#define RED_WEIGHT 0.2989f
#define GREEN_WEIGHT 0.5870f
#define BLUE_WEIGHT 0.1140f
#define MAX_PIXEL(r, g, b) fmaxf(r, fmaxf(g, b))
#define MIN_PIXEL(r, g, b) fminf(r, fminf(g, b))
#define BLOCK_DIM 32

__device__ uchar3 hsv(float3 *rgb)
{
    float h, s, v;
    float r = rgb->x, g = rgb->y, b = rgb->z;
    v = MAX_PIXEL(r, g, b);
    float min = MIN_PIXEL(r, g, b);
    min = v - min;
    s = v != 0.0f ? 255.0f*min / v : 0.0f;

    float tmp = 60.0f / min;
    if (v == r)
        h = tmp*(g - b);
    if (v == g)
        h = 120.0f + tmp*(b - r);
    if (v == b)
        h = 240.0f + tmp*(r - b);
    h = h < 0.0f ? 360.0f + h : h;
    //h = h > 180.0f ? h : 180.0f;

    return make_uchar3((unsigned char)h, (unsigned char)s, (unsigned char)v);
}

__global__ void rgb2hsv(unsigned char *d_input, unsigned char *d_output,int height, int width)
{
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x*blockIdx.x;

    __shared__ float smem[6][32 * 3];
    __shared__ unsigned char hsv_result[6][32*3];
    for (int i = row; i < height; i += blockDim.y*gridDim.y)
        for (int j = col; j + threadIdx.x < width; j += blockDim.x*gridDim.x)
        {
            int index = 3 * (i*width + j) / 4 + threadIdx.x;
            uchar4 p0;
            if (threadIdx.x < 24) // 24 * 4 = 32 * 3
            {
                p0 = reinterpret_cast<uchar4*>(d_input)[index];
                reinterpret_cast<float4*>(smem)[24 * threadIdx.y + threadIdx.x] = make_float4((float)p0.x, (float)p0.y, (float)p0.z, (float)p0.w);
            }
            __syncthreads();

            float3 gray = make_float3(smem[threadIdx.y][3 * threadIdx.x], smem[threadIdx.y][3 * threadIdx.x + 1], smem[threadIdx.y][3 * threadIdx.x + 2]);
            reinterpret_cast<uchar3*>(hsv_result)[32 * threadIdx.y + threadIdx.x] = hsv(&gray);
            __syncthreads();

            if (threadIdx.x < 24) // 24 * 4 = 32 * 3
            {
                uchar4 p1 = reinterpret_cast<uchar4*>(hsv_result)[24 * threadIdx.y + threadIdx.x];
                reinterpret_cast<uchar4*>(d_output)[index] = p1;
            }
        }
}

__device__ uchar3 hsv_(uchar3 *rgb)
{
    float h, s, v;
    float r = rgb->x, g = rgb->y, b = rgb->z;
    v = MAX_PIXEL(r, g, b);
    float min = MIN_PIXEL(r, g, b);
    min = v - min;
    s = v != 0.0f ? 255.0f*min / v : 0.0f;

    float tmp = 60.0f / min;
    if (v == r)
        h = tmp*(g - b);
    if (v == g)
        h = 120.0f + tmp*(b - r);
    if (v == b)
        h = 240.0f + tmp*(r - b);
    h = h < 0.0f ? 360.0f + h : h;
    //h = h > 180.0f ? h : 180.0f;

    return make_uchar3((unsigned char)h, (unsigned char)s, (unsigned char)v);
}

__global__ void rgb2hsv_(unsigned char *d_input, unsigned char *d_output,int height, int width)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i = row; i < height; i += blockDim.y*gridDim.y)
        for (int j = col; j < width; j += blockDim.x*gridDim.x)
        {
            uchar3 p0 = reinterpret_cast<uchar3*>(d_input)[i*width + j];
            reinterpret_cast<uchar3*>(d_output)[i*width + j] = hsv_(&p0);
        }
}


}