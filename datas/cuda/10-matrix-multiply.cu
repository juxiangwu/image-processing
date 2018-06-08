extern "C" {
/*
    #define BLOCK_SIZE 16

    __global__ void matrix_multiply(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n)
    {
        __shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
        const int tidc = threadIdx.x;
        const int tidr = threadIdx.y;
        const int bidc = blockIdx.x * BLOCK_SIZE;
        const int bidr = blockIdx.y * BLOCK_SIZE;
        int i, j;

        float results = 0;
        float comp = 0;

        for(j = 0; j < n; j += BLOCK_SIZE) {
            matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
            matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];

            __syncthreads();

            for(i = 0; i < BLOCK_SIZE; i++) {
                float t;
                comp -= matA[tidr][i] * matB[i][tidc];
                t = results - comp;
                comp = (t - results) + comp;
                results = t;
            }

            __syncthreads();
        }

        c[(tidr + bidr) * ldc + tidc + bidc] = results;
    }
*/

    __global__ void matrix_multiply_0(const float* _A,const float *_B,float* _C,int _wa,int _wb)
    {
        float sum = 0;
        //找出该线程所在的行列
        int row = blockIdx.y*blockDim.y + threadIdx.y;  // X 对应矩阵row, Y对应举证col
        int col = blockIdx.x*blockDim.x + threadIdx.x;

        //线程Thread(row,col)负责计算C(row,col)
        for (int i = 0; i < _wa; ++i)
        {
            sum += _A[row*_wa + i]*_B[i*_wb + col];
        }
        _C[row*_wb + col] = sum;
    }
  
      __global__ void matrix_multiply_1(float *A, float *B, float *C, int numARows,
                                   int numAColumns, int numBRows, int numBColumns,
                                   int numCRows, int numCColumns) {
      //@@ Insert code to implement matrix multiplication here
         float sum = 0.0f;

        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;


        if(row < numCRows && col < numCColumns){
            for (int i = 0; i < numAColumns; ++i)
            {
                sum += A[row*numAColumns + i] * B[i*numBColumns + col];
            }
            C[row*numBColumns + col] = sum;
        }
        //printf("C = %f\n",C[row*numBColumns + col]);

    }
    
    __global__ void matrix_elementwise_multiply(float * A,float * B,float *C,int width,int height){
    
        int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = xIndex + yIndex * width;
        if(xIndex < width && yIndex < height){
            C[idx] = A[idx] * B[idx];
        }
    }
}