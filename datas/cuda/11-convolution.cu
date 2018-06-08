extern "C" {
    //#define Mask_width  5  
    //#define Mask_radius Mask_width/2  
    #define O_TILE_WIDTH 12  
    #define BLOCK_WIDTH (O_TILE_WIDTH+4)  
    #define clamp(x, start, end)    min(max(x, start), end)
    __global__ void convolution_2D_kernel(float*P,float*N,int height,int width,int channels,const  float* __restrict__ M,int Mask_width){
        int Mask_radius = Mask_width/2;
        __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH*3]; 
        int tx=threadIdx.x;  
        int ty=threadIdx.y; 
        int row_o=blockIdx.y*O_TILE_WIDTH+ty;
        int col_o=blockIdx.x*O_TILE_WIDTH+tx;
        
        int row_i=row_o-2;
        int col_i=col_o-2;
        
        int i=0;  
        int j=0;  
        int k=0;
        if((row_i>=0)&&(row_i<height)&&(col_i>=0)&&(col_i<width)){
            for(k=0;k<channels;++k){
                Ns[ty][tx*channels+k]=P[(row_i*width+col_i)*channels+k];
            }
        }else{
            for(k=0;k<channels;++k){
                Ns[ty][tx*channels+k]=0.0f;
            }
        }
        
       __syncthreads();
       float output=0.0f;
       if(ty<O_TILE_WIDTH&&tx<O_TILE_WIDTH){
          for(k=0;k<channels;++k){
           output=0.0f;
           for(i=0;i<Mask_width;++i){
            for(j=0;j<Mask_width;++j){
              output+=M[i*Mask_width+j]*Ns[i+ty][(j+tx)*channels+k];  
            }   
           }
           if(row_o<height&&col_o<width){
             N[(row_o*width+col_o)*channels+k]=output;  
           }
          } 
       }
    }
    
   

}