extern "C" {

    // useful defines
    #define Mask_width 5
    #define Mask_radius Mask_width / 2
    #define TILE_WIDTH 16
    #define w (TILE_WIDTH + Mask_width - 1)
    #define clamp(x) (min(max((x), 0.0), 1.0))

    __global__ void convolution(float *I, const float *__restrict__ M, float *P,
            int channels, int width, int height) {
        __shared__ float N_ds[w][w];
        int k;
        for (k = 0; k < channels; k++) {
            // First batch loading
            int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
            int destY = dest / w;
            int destX = dest % w;
            int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
            int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
            int src = (srcY * width + srcX) * channels + k;
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                N_ds[destY][destX] = I[src];
            } else {
                N_ds[destY][destX] = 0;
            }

            // Second batch loading
            dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
            destY = dest / w;
            destX = dest % w;
            srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
            srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
            src = (srcY * width + srcX) * channels + k;
            if (destY < w) {
                if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                    N_ds[destY][destX] = I[src];
                } else {
                    N_ds[destY][destX] = 0;
                }
            }
            __syncthreads();

            float accum = 0;
            int y, x;
            for (y = 0; y < Mask_width; y++) {
                for (x = 0; x < Mask_width; x++) {
                    accum += N_ds[threadIdx.y + y][threadIdx.x + x]
                            * M[y * Mask_width + x];
                }
            }
            y = blockIdx.y * TILE_WIDTH + threadIdx.y;
            x = blockIdx.x * TILE_WIDTH + threadIdx.x;
            if (y < height && x < width)
                P[(y * width + x) * channels + k] = clamp(accum);
            __syncthreads();
        }
}

}