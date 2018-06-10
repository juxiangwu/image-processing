extern "C" {

    __global__ void pyrup_rgb_kernel(unsigned char *d_in,unsigned char *d_out,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int color_tid = (xIndex)* aabhas + (3 * (yIndex));
        const int color_tid1= (xIndex/2)* colorWidthStep + (3 * (yIndex/2));
        if(yIndex >=width || xIndex>=height)
        {
            return;
        }
       
        if(yIndex%2==0 &&xIndex%2==0)
        {
            d_out[color_tid]=d_in[color_tid1];
            d_out[color_tid+1]=d_in[color_tid1+1];
            d_out[color_tid+2]=d_in[color_tid1+2];
        }
        else
        {
            d_out[color_tid]=0;
            d_out[color_tid+1]=0;//d_in[color_tid1+1];
            d_out[color_tid+2]=0;//d_in[color_tid1+2];

        }
    }
    
    __global__ void pyrup_gray_kernel(unsigned char *d_in,unsigned char *d_out,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int color_tid = (xIndex)* aabhas + yIndex;
        const int color_tid1= (xIndex/2)* colorWidthStep + yIndex/2;
        if(yIndex >=width || xIndex>=height)
        {
            return;
        }
       
        if(yIndex%2==0 &&xIndex%2==0)
        {
            d_out[color_tid]=d_in[color_tid1];
            //d_out[color_tid+1]=d_in[color_tid1+1];
            //d_out[color_tid+2]=d_in[color_tid1+2];
        }
        else
        {
            d_out[color_tid]=255;
            //d_out[color_tid+1]=0;//d_in[color_tid1+1];
            //d_out[color_tid+2]=0;//d_in[color_tid1+2];

        }
    }    
    
    
  __global__ void pyrdown_rgb_kernel(unsigned char *d_in,unsigned char *d_out,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int color_tid = (xIndex)* aabhas + (3 * (yIndex));
        const int color_tid1= (2*xIndex)* colorWidthStep + (3 * (2*yIndex));
        if(yIndex >=width || xIndex>=height)
        {

            return;
        }

        d_out[color_tid]=d_in[color_tid1];
        d_out[color_tid+1]=d_in[color_tid1+1];
        d_out[color_tid+2]=d_in[color_tid1+2];
    }
    
   __global__ void pyrdown_gray_kernel(unsigned char *d_in,unsigned char *d_out,int colorWidthStep,int aabhas,int height,int width)
    {
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
        const int color_tid = (xIndex)* aabhas + yIndex;
        const int color_tid1= (2*xIndex)* colorWidthStep + 2*yIndex;
        if(yIndex >=width || xIndex>=height)
        {

            return;
        }

        d_out[color_tid]=d_in[color_tid1];
        //d_out[color_tid+1]=d_in[color_tid1+1];
        //d_out[color_tid+2]=d_in[color_tid1+2];
    }

}