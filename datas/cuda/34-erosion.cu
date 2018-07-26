extern "C" {
/**
 * Naive erosion kernel with each thread processing a square area.
 */
__global__ void NaiveErosionKernel(int * src, int * dst, int width, int height, int radio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    unsigned int start_i = max(y - radio, 0);
    unsigned int end_i = min(height - 1, y + radio);
    unsigned int start_j = max(x - radio, 0);
    unsigned int end_j = min(width - 1, x + radio);
    int value = 255;
    for (int i = start_i; i <= end_i; i++) {
        for (int j = start_j; j <= end_j; j++) {
            value = min(value, src[i * width + j]);
        }
    }
    dst[y * width + x] = value;
}

/**
 * Two steps erosion using separable filters
 */
__global__ void ErosionStep2(int * src, int * dst, int width, int height, int radio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    unsigned int start_i = max(y - radio, 0);
    unsigned int end_i = min(height - 1, y + radio);
    int value = 255;
    for (int i = start_i; i <= end_i; i++) {
        value = min(value, src[i * width + x]);
    }
    dst[y * width + x] = value;
}
__global__ void ErosionStep1(int * src, int * dst, int width, int height, int radio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) {
        return;
    }
    unsigned int start_j = max(x - radio, 0);
    unsigned int end_j = min(width - 1, x + radio);
    int value = 255;
    for (int j = start_j; j <= end_j; j++) {
        value = min(value, src[y * width + j]);
    }
    dst[y * width + x] = value;
}

/**
 * Two steps erosion using separable filters with shared memory.
 */
__global__ void ErosionSharedStep2(int * src,int * dst, int width, int height,int radio,  int tile_w, int tile_h) {
    __shared__ int smem[32];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * blockDim.x + tx] = 255;
    __syncthreads();
    if (x >= width || y < 0 || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h)) {
        return;
    }
    int * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    int val = smem_thread[0];
    for (int yy = 1; yy <= 2 * radio; yy++) {
        val = min(val, smem_thread[yy * blockDim.x]);
    }
    dst[y * width + x] = val;
}

__global__ void ErosionSharedStep1(int * src, int * dst, int width, int height, int radio, int tile_w, int tile_h) {
    __shared__ int smem[32];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = 255;
    __syncthreads();
    if (x < 0 || x >= width || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) {
        return;
    }
    int * smem_thread = &smem[ty * blockDim.x + tx - radio];
    int val = smem_thread[0];
    for (int xx = 1; xx <= 2 * radio; xx++) {
        val = min(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

}