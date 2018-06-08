
extern "C" {
    //图像平滑
    __global__ void smoothingFilter(int Lx, int Ly, int Threshold, int MaxRad, 
            float* IMG, float* BOX, float* NORM)
        {
        // Indexing
        int tid = threadIdx.x;
        int tjd = threadIdx.y;
        int i = blockIdx.x * blockDim.x + tid;
        int j = blockIdx.y * blockDim.y + tjd;
        int stid = tjd * blockDim.x + tid;
        int gtid = j * Ly + i;  

        // Smoothing params
        float qq    = 1.0;
        float sum   = 0.0;
        float ksum  = 0.0;
        float ss    = qq;
        // Shared memory
        extern __shared__ float s_IMG[];
        s_IMG[stid] = IMG[gtid];
        __syncthreads();

        // Compute all pixels except for image border
        if ( i >= 0 && i < Ly && j >= 0 && j < Lx )
        {
            // Continue until parameters are met
            while (sum < Threshold && qq < MaxRad)
            {
                ss = qq;
                sum = 0.0;
                ksum = 0.0;

                // Normal adaptive smoothing (w/o gaussian sum)
                for (int ii = -ss; ii < ss+1; ii++)
                {
                    for (int jj = -ss; jj < ss+1; jj++)
                    {
                        if ( (i-ss >= 0) && (i+ss < Lx) && (j-ss >= 0) && (j+ss < Ly) )
                        {
                             // Compute within bounds of block dimensions
                            if( tid-ss > 0 && tid+ss < blockDim.x && tjd-ss > 0 && tjd+ss < blockDim.y )
                            {
                                sum += s_IMG[stid + ii*blockDim.y + jj];
                                ksum += 1.0;          
                            }
                            // Compute block borders with global memory
                            else
                            {
                                sum += IMG[gtid + ii*Ly + jj];
                                ksum += 1.0;                   
                            }                                     
                        }
                    }
                }
                qq += 1;
            }
            BOX[gtid] = ss;
            __syncthreads();
            // Determine the normalization for each box
            for (int ii = -ss; ii < ss+1; ii++)
            {
                for (int jj = -ss; jj < ss+1; jj++)
                {
                    if (ksum != 0)
                    {
                        NORM[gtid + ii*Ly + jj] +=  1.0 / ksum;
                    }
                }
            }
        }
        __syncthreads();
        }
        
     __global__ void normalizeFilter(int Lx, int Ly, float* IMG, float* NORM )
        {
        // Indexing
        int tid = threadIdx.x;
        int tjd = threadIdx.y;
        int i = blockIdx.x * blockDim.x + tid;
        int j = blockIdx.y * blockDim.y + tjd;
        int stid = tjd * blockDim.x + tid;
        int gtid = j * Ly + i;  
        // shared memory for IMG and NORM
        extern __shared__ float s_NORM[];
        s_NORM[stid] = NORM[gtid];
        __syncthreads();    
        // Compute all pixels except for image border
        if ( i >= 0 && i < Ly && j >= 0 && j < Lx )
        {
            // Compute within bounds of block dimensions
            if( tid > 0 && tid < blockDim.x && tjd > 0 && tjd < blockDim.y )
            {
                if (s_NORM[stid] != 0)
                {
                    IMG[gtid] /= s_NORM[stid];
                }
            }
            // Compute block borders with global memory
            else
            {
                if (NORM[gtid] != 0)
                {        
                    IMG[gtid] /= NORM[gtid];
                }
            }
        }
        __syncthreads();
        }
        
     __global__ void outFilter( int Lx, int Ly, float* IMG, float* BOX, float* OUT )
        {
        // Indexing
        int tid = threadIdx.x;
        int tjd = threadIdx.y;
        int i = blockIdx.x * blockDim.x + tid;
        int j = blockIdx.y * blockDim.y + tjd;
        int stid = tjd * blockDim.x + tid;
        int gtid = j * Ly + i;  
        // Smoothing params
        float ss    = BOX[gtid];
        float sum   = 0.0;
        float ksum  = 0.0;
        extern __shared__ float s_IMG[];
        s_IMG[stid] = IMG[gtid];
        __syncthreads();
        // Compute all pixels except for image border
        if ( i >= 0 && i < Ly && j >= 0 && j < Lx )
        {
            for (int ii = -ss; ii < ss+1; ii++)
            {
                for (int jj = -ss; jj < ss+1; jj++)
                {
                if ( (i-ss >= 0) && (i+ss < Lx) && (j-ss >= 0) && (j+ss < Ly) )
                    {
                         // Compute within bounds of block dimensions
                        if( tid-ss > 0 && tid+ss < blockDim.x && tjd-ss > 0 && tjd+ss < blockDim.y )
                        {
                            sum += s_IMG[stid + ii*blockDim.y + jj];
                            ksum += 1.0;          
                        }
                        // Compute block borders with global memory
                        else
                        {
                            sum += IMG[gtid + ii*Ly + jj];
                            ksum += 1.0;                   
                        }
                    }
                }
            }
        }  
        if ( ksum != 0 )
        {
            OUT[gtid] = sum / ksum;
        }
        __syncthreads();
        }

}