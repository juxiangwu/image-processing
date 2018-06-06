__global__ void conv(const float *A, const float *B, int aw, int ah, int bw, int bh, int b_sum, float *C){

    /*Get row and column to operate on from thread coordinates*/
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;
    
    /*Calculate "padding" radius of convolution kernel (distance around central pixel)*/
    int pw = (bw-1)/2;
    int ph = (bh-1)/2;

    /*If within the range of C (ie A - padding)*/
    if( row < (ah-2*ph) && col < (aw-2*pw) ) {
        
        /*Set initial pixel value*/
        int val = 0;
        
         /*For each vertical position on the kernel matrix, relative to the central pixel*/
        for(int i=-ph; i<=ph; i=i+1){
            /*Calculate zero-indexed row ID on kernel matrix*/
            int b_row = i+ph; 

            /*For each horizontal position on the kernel matrix, relative to the central pixel*/
            for(int j=-pw; j<=pw; j=j+1){
                /*Calculate zero-indexed column ID on kernel matrix*/
                int b_col = j+pw;

                /*Add product of kernel value and corresponding image value to running total*/
                val += A[ (row+ph +i)*aw + (col+pw +j) ] * B[ b_row*bw + b_col ];
            }
        }
        
        /*Copy appropriately normalised resulting pixel value to position on C matrix*/
        C[row*(aw-2*pw) + col] = val/b_sum;
    }
}