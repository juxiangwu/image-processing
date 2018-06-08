extern "C"{

    const int Tile_width = 16;
    // Kernel to Perform median filter operation on a image using Shared memory
    __global__ void median_filter_shared(unsigned char* imaged, unsigned char* outputImaged,int width,int height ){

        __shared__ unsigned char images[Tile_width+2][Tile_width+2]; // Creating a shared memory element for each block
        int row = blockIdx.y * Tile_width + threadIdx.y; //Current operating row
        int col = blockIdx.x * Tile_width + threadIdx.x; // Current operating colum
        int x = threadIdx.x; // indicates the thread id in the x direction within the block, in other words x indicates the current column within the block
        int y = threadIdx.y; // indicates the thread id in the y direction within the block, in other words y indicates the current row within the block
        unsigned char temp[9];

        if(row<height && col < width){ 

        // Making all borders in shared memory element  to be zero
        if( x == 0 ){ // 
            images [y+1][x] = 0;
            if( y == 0)
                images[y][x] = 0;
            if(y == Tile_width -1)
                images[y+2] [x]= 0;
        }

        if(x == Tile_width -1){
            images[x+2][y+1] = 0;
            if(y==0)
                images[y][x+2] = 0;
            if(y == 15)
                images[y+2][x+2] = 0;
        }

        if( y==0 )
        {
            images[y] [x+1]= 0;
        }

        if( y==15)
        {
            images[y+2][x+1]=0;
        }

        __syncthreads();

        images[y+1][x+1] = imaged[row*width + col]; // Copies the respective elemnts from global memory to shared memory.


        __syncthreads();

        // The following set of code below copies the respective border elements for each shared memory variable from the Global memory.
        if( x == 0 && col>0 && col < width ){

            images [y+1] [x]= imaged[row *width +(col -1)]; // Copies elements to row 1 to 16 in column 0 in shared memory variable from Global memory [w.r.t 3 x 3 filter and Tile_width=16]. 
            if( y == 0)
                images[y] [x]= imaged[(row-1)*width + (col-1)]; // Copies element to the row 0 in column 0 in shared memory variable from global memory[w.r.t 3 x 3 filter and Tile_width=16] .
            if(y == Tile_width -1)
                images[y+2][x] = imaged[(row+1)*width + (col-1)] ;// Copies element to the row 17 in column 0 in shared memory variable from global memory[w.r.t 3 x 3 filter and Tile_width=16].
        }

        if(x == Tile_width -1  && col>0 && col < width ){
            images[y+1][x+2] = imaged[row * width + (col+1)]; // Copies elements to row 1 to 16 in column 17 in shared memory variable from Global memory[w.r.t 3 x 3 filter and Tile_width=16]. 
            if(y==0)
                images[y] [x+2]= imaged[(row-1) * width + (col+1)];// Copies element to the row 0 in column 17 in shared memory variable from global memory[w.r.t 3 x 3 filter and Tile_width=16].
            if(y == 15)
                images[y+2][x+2] = imaged[(row+1) * width + (col+1)];// Copies element to the row 17 in column 17 in shared memory variable from global memory[w.r.t 3 x 3 filter and Tile_width=16].
                }

        if( y==0 && row >0 && row < height)
        {
            images[y][x+1] = imaged[(row-1) * width + col];// Copies elements to col 1 to 16 in row 0 in shared memory variable from Global memory[w.r.t 3 x 3 filter and Tile_width=16]. 
        }
        if( y==15 && row >0 && row < height )
        {
            images[y+2][x+1]= imaged[(row+1) * width + col];// Copies elements to col 1 to 16 in row 17 in shared memory variable from Global memory[w.r.t 3 x 3 filter and Tile_width=16]. 
        }

        __syncthreads();



    // Copies the filter values for a pixels

        temp[0] = images[y][x];
        temp[1] =images [y+1][x];
        temp[2] = images[y+2][x];
        temp[3] = images[y][x+1];
        temp[4] = images[y+1][x+1];
        temp[5] = images[y+2][x+1];
        temp[6] = images [y][x+2];
        temp[7] = images[y+1][x+2];
        temp[8] = images[y+2][x+2];


        __syncthreads();
        // Replication of border pixels
        if(row == 0 || row == height-1 || col == 0 || col == width -1)
        {
        for(int i=0; i < sizeof(temp); i++)
        {

                temp[i]=imaged[row*width + col];
        }
        }

        //Bubble sort to find the median value

        for (int k = 0; k < sizeof(temp); k++) {
          for (int l = k+1; l < sizeof(temp); l++) {
             if (temp[k] > temp[l]) {
               unsigned char temp1 = temp[k];
                temp[k] = temp[l];
                temp[l] = temp1;
                     }
                }
            }
        outputImaged[row * width +col] = temp[4];

        }

    }

    // Kernel to Perform median filter operation on a image using Global memory
    __global__ void median_filter_global(unsigned char* imaged, unsigned char* outputImaged,int width,int height ){


        // Calculates the row and column indices of matrices
        int row = blockIdx.y * Tile_width + threadIdx.y;
        int col = blockIdx.x * Tile_width + threadIdx.x;
        unsigned char temp[9]; // Storing of filter values


        if(row<height && col < width){ // Limits the operating range within the range of image

        if( (col ==0) || (row == 0) || (col == width-1) || (row == height-1)) // Replication of pixels for border conditions
        {
            for (int i=0; i<sizeof(temp);i++)
                temp[i] = imaged[col+width*row];
        }

        // Finding the filter values for non border conditions.
        else{

        temp[0] = imaged[(col-1)+ width*(row-1)];
        temp[1] =imaged[(col-1)+ width*row];
        temp[2] = imaged[(col-1)+ width*(row+1)];
        temp[3] =imaged[(col)+ width*(row-1)];
        temp[4] = imaged[(col)+ width*row];
        temp[5] = imaged[col+ width*(row+1)];
        temp[6] = imaged[(col+1)+ width*(row-1)];
        temp[7] = imaged[(col+1)+ width*row];
        temp[8] = imaged[(col+1)+ width*(row+1)];

        }

        // Bubble sort for finding median value
        for (int k = 0; k < sizeof(temp); k++) {
          for (int l = k+1; l < sizeof(temp); l++) {
             if (temp[k] > temp[l]) {
               unsigned char temp1 = temp[k];
                temp[k] = temp[l];
                temp[l] = temp1;
                     }
                }
            }
            outputImaged[row * width +col] = temp[4]; // Median value is copied to the output image.
        }
    }

}