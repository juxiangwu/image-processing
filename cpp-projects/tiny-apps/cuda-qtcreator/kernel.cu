#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <glm/glm.hpp>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void kernel1(int *a,int *b,int *c) {
    int tid=threadIdx.x + blockIdx.x*blockDim.x;

    if(tid<32)
        c[tid] = a[tid]+b[tid];
}

extern "C"
int cuda_main()
{

    int *h_a,*d_a,*h_b,*d_b,*h_c,*d_c;
    h_a=new int[32];
    h_b=new int[32];
    h_c=new int[32];

    for(int i=0;i<32;i++) {
        h_a[i] = i;
        h_b[i] = i*2;
    }

    cudaMalloc((void**)&d_a,sizeof(int)*32);
    cudaMalloc((void**)&d_b,sizeof(int)*32);
    cudaMalloc((void**)&d_c,sizeof(int)*32);

    cudaMemcpy(d_a,h_a,32*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,32*sizeof(int),cudaMemcpyHostToDevice);

    dim3 blocks(1);
    dim3 threads(32);
    kernel1 <<< blocks,threads >>>(d_a,d_b,d_c);

    cudaMemcpy(h_c,d_c,32*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<32;i++)
        cout << h_a[i] << '+' << h_b[i] << '=' << h_c[i] << endl;


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
