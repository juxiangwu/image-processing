extern "C" {
    #define FLT_MIN 1.175494351e-38F 
    #define FLT_MAX 3.402823466e+38F 
    __global__
    void separateChannels(const uchar4* const inputImageRGBA,
                          int numRows,
                          int numCols,
                          float* const redChannel,
                          float* const greenChannel,
                          float* const blueChannel)
    {
      int absolute_image_position_x = blockDim.x * blockIdx.x + threadIdx.x;
      int absolute_image_position_y = blockDim.y * blockIdx.y + threadIdx.y;

      if ( absolute_image_position_x >= numCols || absolute_image_position_y >= numRows )
        return ;

      int thread_1D_pos = absolute_image_position_y * numCols + absolute_image_position_x;

      redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
      greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
      blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
    }

    __global__
    void rgb_to_xyY(
        float* d_r,
        float* d_g,
        float* d_b,
        float* d_x,
        float* d_y,
        float* d_log_Y,
        float  delta,
        int num_pixels_y,
        int num_pixels_x )
    {
        int  ny = num_pixels_y;
      int  nx = num_pixels_x;
      int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
      int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

      if ( image_index_2d.x < nx && image_index_2d.y < ny )
      {
        float r = d_r[ image_index_1d ];
        float g = d_g[ image_index_1d ];
        float b = d_b[ image_index_1d ];

        float X = ( r * 0.4124f ) + ( g * 0.3576f ) + ( b * 0.1805f );
        float Y = ( r * 0.2126f ) + ( g * 0.7152f ) + ( b * 0.0722f );
        float Z = ( r * 0.0193f ) + ( g * 0.1192f ) + ( b * 0.9505f );

        float L = X + Y + Z;
        float x = X / L;
        float y = Y / L;

        float log_Y = log10f( delta + Y );

        d_x[ image_index_1d ]     = x;
        d_y[ image_index_1d ]     = y;
        d_log_Y[ image_index_1d ] = log_Y;
      }
    }

    __global__ void kernel_scan(int* d_bins, int size)
    {
        int index = blockDim.x*blockIdx.x+threadIdx.x;

        if(index >= size)
          return;
        int temp;
        if(index > 0)
        {
          temp = d_bins[index - 1];
        }
        else
        {
          temp = 0;
        }
        __syncthreads();

        d_bins[index] = temp;
        __syncthreads();

        int val = 0;
        for(int s=1; s<=size; s*=2)
        {
            int a = index-s;
            val = 0; 
            if(a>=0)
                 val = d_bins[a];
            __syncthreads();

            if(a>=0)
                d_bins[index] += val;
            __syncthreads();
        }
    }

    __global__ void kernel_histo(const float* d_in, int* d_bins, float min,float max,int size, int numBins)
    {
        int index = blockDim.x*blockIdx.x+threadIdx.x;
      if(index<size)
      {
          int a = ((d_in[index] - min)/(max-min))* numBins;
          atomicAdd(&d_bins[a], 1);
      }
    }

    __global__ void kernel_maxmin(float* d_in, float*d_out, int size, int maxmin)
    {
        int tid = threadIdx.x;
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        extern __shared__ float shared[];

        if(x>=size)
          return ;

        if(x<size)
            shared[tid] = d_in[x];
        else
        {
            if(maxmin == 0)
                shared[tid] = FLT_MAX;
            else
                shared[tid] = -FLT_MAX;
        }
        __syncthreads();

        for(int s=1; s<blockDim.x; s++)
        {
          if(tid % (2*s) == 0)
          {
            if(s+tid < blockDim.x)
                if(maxmin == 0)
                  shared[tid] = min(shared[tid], shared[tid+s]);
                else
                  shared[tid] = max(shared[tid], shared[tid+s]);

          }
            __syncthreads();
        }
        __syncthreads();

        if(tid == 0)
            d_out[blockIdx.x] = shared[0];
    }


    __global__ void tonemap(
        float* d_x,
        float* d_y,
        float* d_log_Y,
        float* d_cdf_norm,
        float* d_r_new,
        float* d_g_new,
        float* d_b_new,
        float  min_log_Y,
        float  max_log_Y,
        float  log_Y_range,
        int    num_bins,
        int    num_pixels_y,
        int    num_pixels_x )
    {
      int  ny             = num_pixels_y;
      int  nx             = num_pixels_x;
      int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
      int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

      if ( image_index_2d.x < nx && image_index_2d.y < ny )
      {
        float x         = d_x[ image_index_1d ];
        float y         = d_y[ image_index_1d ];
        float log_Y     = d_log_Y[ image_index_1d ];
        int   bin_index = min( num_bins - 1, int( (num_bins * ( log_Y - min_log_Y ) ) / log_Y_range ) );
        float Y_new     = d_cdf_norm[ bin_index ];

        float X_new = x * ( Y_new / y );
        float Z_new = ( 1 - x - y ) * ( Y_new / y );

        float r_new = ( X_new *  3.2406f ) + ( Y_new * -1.5372f ) + ( Z_new * -0.4986f );
        float g_new = ( X_new * -0.9689f ) + ( Y_new *  1.8758f ) + ( Z_new *  0.0415f );
        float b_new = ( X_new *  0.0557f ) + ( Y_new * -0.2040f ) + ( Z_new *  1.0570f );

        d_r_new[ image_index_1d ] = r_new;
        d_g_new[ image_index_1d ] = g_new;
        d_b_new[ image_index_1d ] = b_new;
      }
    }

    __global__
    void recombineChannels(const float* const redChannel,
                           const float* const greenChannel,
                           const float* const blueChannel,
                           uchar4* const outputImageRGBA,
                           int numRows,
                           int numCols)
    {
      const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                            blockIdx.y * blockDim.y + threadIdx.y);

      const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

      if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

      unsigned char red   = redChannel[thread_1D_pos];
      unsigned char green = greenChannel[thread_1D_pos];
      unsigned char blue  = blueChannel[thread_1D_pos];

      //Alpha should be 255 for no transparency
      uchar4 outputPixel = make_uchar4(red, green, blue, 255);

      outputImageRGBA[thread_1D_pos] = outputPixel;
    }

    __global__ void normalize_cdf(
        unsigned int* d_input_cdf,
        float*        d_output_cdf,
        int           n
        )
    {
      const float normalization_constant = 1.f / d_input_cdf[n - 1];

      int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

      if ( global_index_1d < n )
      {
        unsigned int input_value  = d_input_cdf[ global_index_1d ];
        float        output_value = input_value * normalization_constant;

        d_output_cdf[ global_index_1d ] = output_value;
      }
    }

}