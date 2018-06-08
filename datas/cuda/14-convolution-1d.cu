extern "C"{

    __global__ void convolution_1D_basic_kernel(int *N, int *M, int *P,  
        int Mask_Width, int Width){  
        int i = blockIdx.x*blockDim.x + threadIdx.x;  

        float Pvalue = 0;  
        int N_start_point = i - (Mask_Width / 2);  
        for (int j = 0; j < Mask_Width; j++){  
            if(N_start_point + j >= 0 && N_start_point + j < Width){  
                Pvalue += N[N_start_point + j] * M[j];  
            }  
        }  
        P[i] = Pvalue;  
    }
}