extern "C" {

    __device__ int Reflect(int size, int p)
    {
        if (p < 0)
            return -p - 1;

        if (p >= size)
            return 2*size - p - 1;
        return p;
    }

    __global__ void convolution_kernel_single_channel(float* src, float* dst, int width, int height, float* kernel, int kernel_width)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        // consider only valid pixel coordinates
        if((x < width) && (y < height))
        {
            // pixel index in the src array
            const int pixel_tid = y * width + x;
            int i, j, x_tmp, y_tmp, flat_index, flat_kernel_index;
            int k = kernel_width / 2;
            float sum = 0.0;

            for (int n = 0; n < kernel_width*kernel_width; n++)
            {
                i = n % kernel_width;
                j = n / kernel_width;

                x_tmp = Reflect(width, x-(j-k));
                y_tmp = Reflect(height, y-(i-k));

                flat_index = x_tmp  + width   * y_tmp ;
                flat_kernel_index = i + kernel_width * j;

                sum += kernel[flat_kernel_index] * src[flat_index];
            }

            dst[pixel_tid] = sum;
        }
    }
    
    __global__ void convolution_rgb(float3* src, float3* dst, int width, int height, float* kernel, int kernel_width)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        // consider only valid pixel coordinates
        if((x < width) && (y < height))
        {
            // pixel index in the src array
            const int pixel_tid = y * width + x;
            int i, j, x_tmp, y_tmp, flat_index, flat_kernel_index;
            int k = kernel_width / 2;
            float sumr = 0.0,sumg=0.0,sumb=0.0;

            for (int n = 0; n < kernel_width*kernel_width; n++)
            {
                i = n % kernel_width;
                j = n / kernel_width;

                x_tmp = Reflect(width, x-(j-k));
                y_tmp = Reflect(height, y-(i-k));

                flat_index = x_tmp  + width   * y_tmp ;
                flat_kernel_index = i + kernel_width * j;

                sumr += kernel[flat_kernel_index] * src[flat_index].x;
                sumg += kernel[flat_kernel_index] * src[flat_index].z;
                sumb += kernel[flat_kernel_index] * src[flat_index].z;
            }

            dst[pixel_tid].x = sumr;
            dst[pixel_tid].y = sumg;
            dst[pixel_tid].z = sumb;
        }
    }

}