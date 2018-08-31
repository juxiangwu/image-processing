/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void increment_kernel(int *g_data,int inc_value){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int* data,const int n,const int x){
    for(int i = 0;i < n;i++){
        if(data[i] !=x){
            printf("Error! data[%d] = %d ,ref = %d\n",i,data[i,x]);
            return false;
        }
    }
    return true;
}

int main(int argc,char * argv[]){
    int devID;
    cudaDeviceProp deviceProps;
    printf("[%s] - Starting...\n",argv[0]);
    devID = findCudaDevice(argc,(const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps,devID));
    printf("CUDA device [%s]\n",deviceProps.name);

    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value = 26;

    //分配HOST内存
    int * a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a,nbytes));
    memset(a,0,nbytes);

    //分配CUDA设备内存
    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a,nbytes));
    checkCudaErrors(cudaMemset(d_a,255,nbytes));

    //设置内核启动配置
    dim3 threads = dim3(512,1);
    dim3 blocks = dim3(n / threads.x,1);

    //创建CUDA事件Handle
    cudaEvent_t start,stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    StopWatchInterface * timer= NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    sdkStartTimer(&timer);
    cudaEventRecord(start,0);
    cudaMemcpyAsync(d_a,a,nbytes,cudaMemcpyHostToDevice,0);

    increment_kernel <<<blocks,threads,0,0>>>(d_a,value);
    cudaMemcpyAsync(a,d_a,nbytes,cudaMemcpyDeviceToHost,0);
    cudaEventRecord(stop,0);
    sdkStopTimer(&timer);

    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady){
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time,start,stop));

    //打印输出时间
    printf("time spent executing by the GPU %.2f\n",gpu_time);
    printf("time spent by CPU in CUDA calls:%.2f\n",sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n",counter);

    bool bFinalResults = correct_output(a, n, value);
    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    cudaDeviceReset();
}