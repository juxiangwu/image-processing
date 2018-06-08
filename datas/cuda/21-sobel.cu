extern "C"{
    const int Tile_width = 16;

    __constant__ double filter_d1[9];//Constant memory variable
    __constant__ double filter_d2[9];//Constant memory variable


    texture<unsigned char,2,cudaReadModeElementType> texIn; // Input to texture memory

    __global__ void sobel_kernel_1(unsigned char* imaged, unsigned char* outputImaged,int width,int height,double* filter1, double* filter2){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0,accum1 = 0,accum2 = 0;
                for(int fw = 0 ; fw < 3; fw++)
                    for(int fh = 0; fh < 3; fh++)
                    {
                        int ix = ( col - 1 + fw + width)% width;
                        int iy = ( row - 1 + fh + height)%height;
                        accum1 = accum1 + (imaged[iy * width + ix] * filter1[fw*3 + fh]);
                        accum2 = accum2 + (imaged[iy * width + ix] * filter2[fw*3 + fh]);

                    }
                accum= sqrt(pow(accum1,2)+pow(accum2,2));
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }

    __global__ void sobel_kernel_2(unsigned char* imaged, unsigned char* outputImaged,int width,int height){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0,accum1 = 0,accum2 = 0;
                for(int fw = 0 ; fw < 3; fw++)
                    for(int fh = 0; fh < 3; fh++)
                    {
                        int ix = ( col - 1 + fw + width)% width;
                        int iy = ( row - 1 + fh + height)%height;
                        accum1 = accum1 + (imaged[iy * width + ix] * filter_d1[fw*3 + fh]);
                        accum2 = accum2 + (imaged[iy * width + ix] * filter_d2[fw*3 + fh]);

                    }
                accum= sqrt(pow(accum1,2)+pow(accum2,2));
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }

    __global__ void sobel_kernel_3(unsigned char* outputImaged,int width,int height,double* filter1, double* filter2){

        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;

        if(row < height && col < width){
                //Perform Image convolution 
            double accum = 0,accum1 = 0,accum2 = 0;
                for(int fw = 0 ; fw < 3; fw++)
                    for(int fh = 0; fh < 3; fh++)
                    {
                        int ix = ( col - 1 + fw + width)% width;
                        int iy = ( row - 1 + fh + height)%height;
                        accum1 = accum1 + (tex2D(texIn,ix,iy) * filter1[fw*3 + fh]);
                        accum2 = accum2 + (tex2D(texIn,ix,iy) * filter2[fw*3 + fh]);

                    }
                accum= sqrt(pow(accum1,2)+pow(accum2,2));
                unsigned char temp = accum;
                outputImaged[row * width + col] = temp;
        }
    }
}