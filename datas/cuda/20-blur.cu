extern "C" {

    const int Tile_width = 16;

    __constant__ double filter_d1[25];//Constant memory variable

    texture<unsigned char,2,cudaReadModeElementType> texIn; // Input to texture memory

    __global__ void blur_kernel(unsigned char* imaged, unsigned char* outputImaged,int width,int height,double* filter){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0;
                for(int fw = 0 ; fw < 5; fw++)
                    for(int fh = 0; fh < 5; fh++)
                    {
                        int ix = ( col - 2 + fw + width)% width;
                        int iy = ( row - 2 + fh + height)%height;
                        accum = accum + (imaged[iy * width + ix] * filter[fw*5 + fh]);
                    }
                accum /= 13;
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }

    __global__ void blur_kernel_1(unsigned char* imaged, unsigned char* outputImaged,int width,int height){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0;
                for(int fw = 0 ; fw < 5; fw++)
                    for(int fh = 0; fh < 5; fh++)
                    {
                        int ix = ( col - 2 + fw + width)% width;
                        int iy = ( row - 2 + fh + height)%height;
                        accum = accum + (imaged[iy * width + ix] * filter_d1[fw*5 + fh]);
                    }
                accum /= 13;
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }

    __global__ void blur_kernel_3( unsigned char* outputImaged,int width,int height,double* filter){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0;
                for(int fw = 0 ; fw < 5; fw++)
                    for(int fh = 0; fh < 5; fh++)
                    {
                        int ix = ( col - 2 + fw + width)% width;
                        int iy = ( row - 2 + fh + height)%height;
                        accum = accum + (tex2D(texIn,ix,iy) * filter[fw*5 + fh]);
                    }
                accum /= 13;
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }

}