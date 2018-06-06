extern "C" {

__global__ void matrix_add(float A[${M}][${N}],float B[${M}][${N}],float C[${M}][${N}]){

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   
   C[x][y] = A[x][y] + B[x][y];
}


}