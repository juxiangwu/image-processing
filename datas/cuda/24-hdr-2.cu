extern "C"{
   __device__ float dot(float3 vec1,float3 vec2){
       float dp = 0.0;
       dp += vec1.x * vec2.x;
       dp += vec1.y * vec2.y;
       dp += vec1.z * vec2.z;
       return dp;
   }
    __device__ float luminance(const float3 color)
    {
        return dot(color, make_float3( 0.2126f, 0.7152f, 0.0722f ));
    }

    __device__ float Uncharted2Tonemap(float x)
    {
        // from http://filmicworlds.com/blog/filmic-tonemapping-operators/
        constexpr float A = 0.15;
        constexpr float B = 0.50;
        constexpr float C = 0.10;
        constexpr float D = 0.20;
        constexpr float E = 0.02;
        constexpr float F = 0.30;
        return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F;
    }

    __device__ float tonemap_f(float c, float exposure)
    {
        constexpr float W = 11.2;
        return Uncharted2Tonemap(c * exposure * 2.0f) / Uncharted2Tonemap(W);
    }

    __device__ float3 tonemap(const float3& c, float exposure)
    {
        return { tonemap_f(c.x, exposure), tonemap_f(c.y, exposure), tonemap_f(c.z, exposure) };
    }

    __device__ unsigned char toLinear8(float c)
    {
        return static_cast<unsigned char>(saturate(c) * 255.0f);
    }

    __device__ unsigned char toSRGB8(float c)
    {
        return toLinear8(powf(c, 1.0f / 2.2f));
    }

    __device__ float fromLinear8(unsigned char c)
    {
        return c * (1.0f / 255.0f);
    }

    __device__ float fromSRGB8(unsigned char c)
    {
        return powf(fromLinear8(c), 2.2f);
    }

    // kernel that computes average luminance of a pixel
    __global__ void luminance_kernel(float* dest, const float* input, unsigned int width, unsigned int height)
    {
        // each thread needs to know on which pixels to work -> get absolute coordinates of the thread in the grid
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        // input is stored as array (row-wise). Load the pixel values
        // offset of first pixel  = y coordinates * width of block (b/c full rows already read) + x coordinates

        if (x < width && y < height){
            const float* input_pixel = input + 3 * (width*y + x); // *3 b/c three colors (=bytes) per pixel.

            float lum =  0.21f * input_pixel[0] + 0.72f * input_pixel[1] + 0.07f * input_pixel[2];  //compensate for human vision

            dest[width *y + x] = lum; //store the results. not * 3 b/c only luminance, no colors
        }
    }

    __global__ void downsample_kernel(float* dest, float* input, unsigned int width, unsigned int height, unsigned int outputPitch, unsigned int inputPitch){
        // each thread needs to know on which pixels to work -> get absolute coordinates for/of the thread in the grid
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

            int F = 2; //width of downsampling square (F^2 = number of pixels to be pooled together)

            //printf(" KERNEL width %d | height %d \n", width, height);
            if ((x*F > width -1) || (y*F > height -1))
                return;

            float sum = 0.0f;

            int nb_counted = F * F;
            // number of pixels to be counted: We start at pixel (x*F, y*F), continue till coordinate(width-1, height-1)
            int xDim = min(F, width  - x*F);
            int yDim = min(F, height - y*F);
            //printf("xDim: %d, yDim: %d \n", xDim, yDim);
            nb_counted = xDim * yDim;
            //printf(" counted %d | width %d | height %d | \t x %d | y %d \n", nb_counted, width, height, x*F, y*F);

            // 2D version: add pixels in 2x2 block, calculate the average. Jump with F so different threads don't operate on the same pixels.
            for (int j = 0; j<F; j++){
                for (int i = 0; i < F; i++){
                    // current pixel : (x*F+i, y*F+j)
                    if ((y*F+j) < height && (x*F+i) < width){ //only sum pixels inside the image, don't count overlapping at the right or bottom sides
                        sum += input[(y*F+j) * inputPitch + x*F +i];
                    }
                }
            }
            dest[ y* outputPitch + x] = sum / nb_counted;
    }

    // first do it on the x direction
    // then do it on the y direction (-> texture memory to make this fast)
    __constant__ float weights[] = {  //weights in one dimension -> 33x33 filter
            //  -16           -15         -14          -13          -12          -11          -10           -9           -8           -7           -6           -5           -4           -3           -2           -1
            0.00288204f, 0.00418319f, 0.00592754f, 0.00819980f, 0.01107369f, 0.01459965f, 0.01879116f, 0.02361161f, 0.02896398f, 0.03468581f, 0.04055144f, 0.04628301f, 0.05157007f, 0.05609637f, 0.05957069f, 0.06175773f,
            //    0
            0.06250444f,
            //    1             2           3            4            5            6            7            8            9           10           11           12           13           14           15           16
            0.06175773f, 0.05957069f, 0.05609637f, 0.05157007f, 0.04628301f, 0.04055144f, 0.03468581f, 0.02896398f, 0.02361161f, 0.01879116f, 0.01459965f, 0.01107369f, 0.00819980f, 0.00592754f, 0.00418319f, 0.00288204f
        };

    __global__ void blur_kernel_x(float* dest, const float* src, unsigned int width, unsigned int height, unsigned int inputPitch, unsigned int outputPitch)
    {
        // 1 thread per output pixel
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    //	__shared__ float blur_buffer[65*65];
    //	if (threadIdx.x == 0){
    //		// read 64x64 input image into shared buffer
    //		// make sure you don't read out of bounds when the kernels operate near the image borders
    //		int startx = max(0,x - 64);
    //		int endx = min(width-1, x+64);
    //
    //		int starty = max(0,y - 64);
    //		int endy = min(height-1, y+64);
    //
    //		blur_buffer = src[startx:endx][starty:endy];
    //		for (int i = 0; i < 65; i++){
    //			for (int j = 0; j< 65){
    //				// if the kernel is near the borders, you need to fill the buffer with zeros at the right places
    //				// so the access of the  buffer is easy relative to the thread coordinates.
    //			}
    //		}
    //	}

        float sumR = 0.0f;
        float sumG = 0.0f;
        float sumB = 0.0f;

        for (int i = -16; i<=16; i++){
            if ((3*(x+i) > 0) && (3*(x+i) < 3*width)){
                sumR += src[3*y*inputPitch + 3*(x+i)] * weights[i+16];
                sumG += src[3*y*inputPitch + 3*(x+i) +1] * weights[i+16];
                sumB += src[3*y*inputPitch + 3*(x+i) +2] * weights[i+16];
            }
        }
    //	if (!threadIdx.x){
    //		printf("sumR: %f | sumG: %f | sumB: %f \n", sumR, sumG, sumB);
    //	}

        dest[ 3*y* outputPitch + 3*x]    = sumR;
        dest[ 3*y* outputPitch + 3*x +1] = sumG;
        dest[ 3*y* outputPitch + 3*x +2] = sumB;
    }



    __global__ void blur_kernel_y(float* dest, const float* src, unsigned int width, unsigned int height, unsigned int inputPitch, unsigned int outputPitch)
    {

        // 1 thread per output pixel
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        float sumR = 0.0f;
        float sumG = 0.0f;
        float sumB = 0.0f;

        for (int i = -16; i<=16; i++){
            if ((3*(y+i) > 0) && (3*(y+i) < 3*height)){
    // all threads in the block read the same pixels. -> lots of overlap.
    // each block accesses 32x32 pixels plus 16 on each side for the gaussian operation -> 64x64 pixels are to be read
    // Before processing, have the threads each read 4 pixels (32x32 threads, 64x64 pixels) in a shared buffer. Then Sync, and have the threads process data from this shared buffer instead of from memory.

                sumR += src[3*(y+i)*inputPitch + 3*x] * weights[i+16];
                sumG += src[3*(y+i)*inputPitch + 3*x +1] * weights[i+16];
                sumB += src[3*(y+i)*inputPitch + 3*x +2] * weights[i+16];
            }
        }

        dest[ 3*y* outputPitch + 3*x]    = sumR;
        dest[ 3*y* outputPitch + 3*x +1] = sumG;
        dest[ 3*y* outputPitch + 3*x +2] = sumB;
    }
    
    __global__ void tonemap_kernel(float* tonemapped, float* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_threshold)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            // figure out input color
            float3 c = { src[3 * (y * width + x) + 0], src[3 * (y * width + x) + 1], src[3 * (y * width + x) + 2] };

            // compute tonemapped color
            float3 c_t = tonemap(c, exposure);

            // write out tonemapped color
            tonemapped[3 * (y * width + x) + 0] = c_t.x;
            tonemapped[3 * (y * width + x) + 1] = c_t.y;
            tonemapped[3 * (y * width + x) + 2] = c_t.z;

            // write out brightpass color
            float3 c_b = luminance(c_t) > brightpass_threshold ? c_t : make_float3(0.0f, 0.0f, 0.0f);
            brightpass[3 * (y * width + x) + 0] = c_b.x;
            brightpass[3 * (y * width + x) + 1] = c_b.y;
            brightpass[3 * (y * width + x) + 2] = c_b.z;
        }
    }
}
