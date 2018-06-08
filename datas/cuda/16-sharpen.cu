extern "C"{

    __global__ void sharpen_kernel(float* curr_im, float* next_im, int height, int width, float epsilon)
    {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      int tid = width*y + x;

      if (y > 0 && y < height - 1 && x > 0 && x < width - 1){
        next_im[tid] = curr_im[tid] + epsilon * 
            (-1*curr_im[tid-width-1] + -2*curr_im[tid-width] + -1*curr_im[tid-width+1]
            + -2*curr_im[tid-1] + 12*curr_im[tid] + -2 * curr_im[tid+1]
            + -1*curr_im[tid+width-1] + -2 * curr_im[tid + width] + -1 * curr_im[tid + width + 1]);
      }
    }

    __global__ void sharpen_rgb_kernel(float3* curr_im, float3* next_im, int height, int width, float epsilon)
    {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      int tid = width*y + x;

      if (y > 0 && y < height - 1 && x > 0 && x < width - 1){
        next_im[tid].x = curr_im[tid].x + epsilon * 
            (-1*curr_im[tid-width-1].x + -2*curr_im[tid-width].x + -1*curr_im[tid-width+1].x
            + -2*curr_im[tid-1].x + 12*curr_im[tid].x + -2 * curr_im[tid+1].x
            + -1*curr_im[tid+width-1].x + -2 * curr_im[tid + width].x + -1 * curr_im[tid + width + 1].x);

        next_im[tid].y = curr_im[tid].y + epsilon * 
            (-1*curr_im[tid-width-1].y + -2*curr_im[tid-width].y + -1*curr_im[tid-width+1].y
            + -2*curr_im[tid-1].y + 12*curr_im[tid].y + -2 * curr_im[tid+1].y
            + -1*curr_im[tid+width-1].y + -2 * curr_im[tid + width].y + -1 * curr_im[tid + width + 1].y);

       next_im[tid].z = curr_im[tid].z + epsilon * 
            (-1*curr_im[tid-width-1].z + -2*curr_im[tid-width].z + -1*curr_im[tid-width+1].z
            + -2*curr_im[tid-1].z + 12*curr_im[tid].z + -2 * curr_im[tid+1].z
            + -1*curr_im[tid+width-1].z + -2 * curr_im[tid + width].z + -1 * curr_im[tid + width + 1].z);

      }
    }

    const int Tile_width = 16;

    __constant__ double filter_d1[9]; //Constant memory variable

    texture<unsigned char,2,cudaReadModeElementType> texIn; // Input to texture memory

    __global__ void sharpen_kernel_1(unsigned char* imaged, unsigned char* outputImaged,int width,int height,double* filter){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0;
                for(int fw = 0 ; fw < 3; fw++)
                    for(int fh = 0; fh < 3; fh++)
                    {
                        int ix = ( col - 1 + fw + width)% width;
                        int iy = ( row - 1 + fh + height)%height;
                        accum = accum + (imaged[iy * width + ix] * filter[fw*3 + fh]);
                    }
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }

    __global__ void sharpen_kernel_2(unsigned char* imaged, unsigned char* outputImaged,int width,int height){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0;
                for(int fw = 0 ; fw < 3; fw++)
                    for(int fh = 0; fh < 3; fh++)
                    {
                        int ix = ( col - 1 + fw + width)% width;
                        int iy = ( row - 1 + fh + height)%height;
                        accum = accum + (imaged[iy * width + ix] * filter_d1[fw*3 + fh]);
                    }
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }
    

    __global__ void sharpen_kernel_3( unsigned char* outputImaged,int width,int height,double* filter){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0;
                for(int fw = 0 ; fw < 3; fw++)
                    for(int fh = 0; fh < 3; fh++)
                    {
                        int ix = ( col - 1 + fw + width)% width;
                        int iy = ( row - 1 + fh + height)%height;
                        accum = accum + (tex2D(texIn,ix,iy) * filter[fw*3 + fh]);
                    }

                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }

}