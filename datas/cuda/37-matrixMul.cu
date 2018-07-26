    extern "C" {

    __global__ void matrixMul(cuComplex* C, cuComplex* A, cuComplex* B, int wA, int wB)
    {
        // Block index
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Index of the first sub-matrix of A processed by the block
        int aBegin = wA * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd = aBegin + wA - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep = BLOCK_SIZE * wB;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        cuComplex Csub;
        Csub.x = 0;
        Csub.y = 0;

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
        {

            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ cuComplex As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ cuComplex Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            AS(ty, tx) = A[a + wA * ty + tx];
            BS(ty, tx) = B[b + wB * ty + tx];

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
            for (int k = 0; k < BLOCK_SIZE; ++k)
                Csub = cuCaddf(Csub, cuCmulf(AS(ty, k), BS(k, tx)));
            //Csub += AS(ty, k) * BS(k, tx);

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        C[c + wB * ty + tx] = Csub;
    }

    __global__ void divide_by_N_azimuth(cufftComplex *d_out, int N)
    {
        //step 1: d_out signal normalization, after cuFFT inverse of d_out from host.

        int thread_ID = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
        if (thread_ID < N)
        {
            d_out[thread_ID].x = d_out[thread_ID].x / (2 * 256);
            d_out[thread_ID].y = d_out[thread_ID].y / (2 * 256);
        }
        __syncthreads();
    }
    __global__ void divide_by_N_range(cufftComplex *d_out, int N)
    {
        //step 1: d_out signal normalization, after cuFFT inverse of d_out from host.

        int thread_ID = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
        if (thread_ID < N)
        {
            d_out[thread_ID].x = d_out[thread_ID].x / (2 * 512);
            d_out[thread_ID].y = d_out[thread_ID].y / (2 * 512);
        }
        __syncthreads();
    }
    __global__ void process_range(cuComplex *tx_new, cuComplex *rx_new, cuComplex *d_out, cuComplex *d_tmp, int N)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

        if (index < N)
        {
            d_out[index] = cuCmulf(d_tmp[0], cuCmulf(cuConjf(tx_new[index]), rx_new[index]));
        }
        __syncthreads();
    }

    __global__ void swap_data(cufftComplex *d_tmp_out, cufftComplex *d_out, int N)
    {
        int thread_ID = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
        if (thread_ID < N)
        {
            if (blockIdx.x % 2 == 0)
            {
                d_tmp_out[thread_ID] = d_out[thread_ID + 512];
            }
            else
            {
                d_tmp_out[thread_ID] = d_out[thread_ID - 512];
            }
        }
        __syncthreads();
    }
    __global__ void process_az(cuComplex *tx_new, cuComplex *rx_new, cuComplex *d_out, int N)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

        if (index < N)
        {
            d_out[index] = cuConjf(cuCmulf(cuConjf(rx_new[index]), tx_new[index]));
        }

        __syncthreads();
    }
    }