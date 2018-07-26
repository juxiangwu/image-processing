/*
 * Created by Rob Golshan
 * Demos common image filters using parallel gpu algorithms
 * Algorithms based of convolutionSeperable.pdf in cuda samples
 * and wjarosz_convolution_2001.pdf --> converted to parallel
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "convolution.cuh"

// Kernel cannot have radius bigger than 15
__constant__ int d_kernel[1024];

#define BLOCK_SIZE 16 

/* 
 * Converts a uint to a uint3, seperating RGB
 * colors by every byte.
 * Most significat to least significant:
 * Red, Green, Blue
 */
__device__ __forceinline__ int3 d_uintToRGB(unsigned int orig)
{
    int3 rgb;
    rgb.x = orig & 0xff;
    rgb.y = (orig>>8)&0xff;
    rgb.z = (orig>>16)&0xff;
    return rgb;
}

/*
 * Converts a uint3 to an unsigned int
 * Assumes each vector member correspond to RGB colors
 * Truncates rgb colors bigger than 1 byte
 */
__device__ __forceinline__ unsigned int d_rgbToUint(int3 rgb)
{
    if (rgb.x > 0xff) rgb.x = 0xff;
    else if (rgb.x < 0) rgb.x = 0;
    if (rgb.y > 0xff) rgb.y = 0xff;
    else if (rgb.y < 0) rgb.y = 0;
    if (rgb.z > 0xff) rgb.z = 0xff;
    else if (rgb.z < 0) rgb.z = 0;

    return (rgb.x & 0xff) | ((rgb.y & 0xff) << 8) | ((rgb.z & 0xff) << 16);
}

/*
 * divides an int3 by an int
 * Maybe faster to just multiply by float instead..
 */
__device__ __forceinline__ int3 d_divide(int3 orig, int op)
{
    orig.x = orig.x/op;
    orig.y = orig.y/op;
    orig.z = orig.z/op;
    return orig;
}

/* The most basic convolution method in parallel
 * Does not take advantage of memory optimizations with a GPU
 * Can be used with any (square) kernel filter
 * SLOW
 * Each output pixel does radius^2 multiplications 
 * T = O(radius^2)
 * W = O(radius^2 * width * height)
 */
__global__ void d_slowConvolution(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius, int weight)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned int loc =  x + y*width;
    int3 accumulation = make_int3(0,0,0);
    int3 value;

    if (x >= width || y >= height) return;
    assert(x < width);
    assert(y < height);
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            if ((x + i < 0) || //left side out of bounds
                (x + i >= width) || //right side OoB
                (y + j < 0) || //top OoB
                (y + j >= height)) //bot OoB
                continue;
            value = d_uintToRGB(d_img[loc + i + j * width]);
            int temp = d_kernel[i + radius +  (j+radius)*((radius << 1) + 1)];
            value *= temp;
            accumulation += value;
        }
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}

/* The most basic convolution method in parallel
 * Takes advantage of shared memory in a GPU 
 * Can be used with any (square) kernel filter
 * Faster than without shared memory 
 * Each output pixel does radius^2 multiplications 
 * T = O(radius^2)
 * W = O(radius^2 * width * height)
 */
__global__ void d_sharedSlowConvolution(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius, int weight)
{
    // Use a 1d array instead of 2D in order to coalesce memory access
    extern __shared__ unsigned int data[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // memory location in d_img
    const unsigned int loc =  x + y*width;

    int3 accumulation = make_int3(0,0,0);
    int3 value;

    int w = blockDim.x;
    int h = blockDim.y;

    /* to convolute the edges of a block, the shared memory must extend outwards of radius  */
#pragma unroll 3 
    for (int i = -w; i <= w; i+= w) {
#pragma unroll 3
        for (int j = -h; j <= h; j+= h) {
            int x0 = threadIdx.x + i;
            int y0 = threadIdx.y + j;
            int newLoc = loc + i + j*width;
            if (x0 < -radius || 
                x0 >= radius + w ||
                y0 < -radius ||
                y0 >= radius + h || 
                newLoc < 0 ||
                newLoc >= width*height)
                continue;
            data[threadIdx.x + i + radius + (threadIdx.y + j + radius)*(blockDim.x+(radius << 1))] = d_img[newLoc];
        }
    }

    __syncthreads();

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            unsigned int t = data[threadIdx.x + i + radius + (threadIdx.y + j + radius)*(blockDim.x+(radius << 1))];
            int temp = d_kernel[i + radius +  (j+radius)*((radius << 1) + 1)];
            value = d_uintToRGB(t);
            value *= temp; 
            accumulation += value;
        }
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}

/* VERY FAST convolution method in parallel 
 * Takes advantage of shared memory in a GPU 
 * Can be used with ONLY WITH SEPARABLE kernel filters
 * Each output pixel does radius+radius multiplications 
 * T = O(radius + radius)
 * W = O(radius * width + radius*height)
 */
__global__ void d_sepRowConvolution(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius)
{
    // Use a 1d array instead of 2D in order to coalesce memory access
    extern __shared__ unsigned int data[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // memory location in d_img
    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;

    int3 accumulation = make_int3(0,0,0);
    int3 value;
    int weight = 0;


    int w = blockDim.x;

    /* to convolute the edges of a block, the shared memory must extend outwards of radius  */
#pragma unroll 3
    for (int i = -w; i <= w; i+= w) {
        int x0 = threadIdx.x + i;
        int newLoc = loc + i;
        if (x0 < -radius || 
            x0 >= radius + w ||
            newLoc < 0 || 
            newLoc >= width*height)
            continue;
        data[threadIdx.x + i + radius + (threadIdx.y) *(blockDim.x+(radius << 1))] = d_img[newLoc];
    }

    __syncthreads();

    for (int i = -radius; i <= radius; i++) {
        unsigned int t = data[threadIdx.x + i + radius + (threadIdx.y)*(blockDim.x+(radius << 1))];
        int temp = d_kernel[i + radius];
        value = d_uintToRGB(t);
        value *= temp;
        weight += temp;
        accumulation += value;
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}

/* VERY FAST convolution method in parallel 
 * Takes advantage of shared memory in a GPU 
 * Can be used with ONLY WITH SEPERABLE kernel filters
 * Each output pixel does radius^2 multiplications 
 * T = O(radius + radius)
 * W = O(radius * width + radius*height)
 */
__global__ void d_sepColConvolution(unsigned int *d_result, int width, int height, int radius)
{
    // Use a 1d array instead of 2D in order to coalesce memory access
    extern __shared__ unsigned int data[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // memory location in d_img
    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;

    int3 accumulation = make_int3(0,0,0);
    int3 value;
    int weight = 0;


    int h = blockDim.y;

    /* to convolute the edges of a block, the shared memory must extend outwards of radius  */
#pragma unroll 3
    for (int j = -h; j <= h; j+= h) {
        int y0 = threadIdx.y + j;
        int newLoc = loc + j*width;
        if (y0 < -radius || 
            y0 >= radius + h ||
            newLoc < 0 ||
            newLoc >= width*height)
            continue;
            data[threadIdx.x + (threadIdx.y + j + radius)*(blockDim.x)] = d_result[newLoc];
    }

    __syncthreads();

    for (int j = -radius; j <= radius; j++) {
        unsigned int t = data[threadIdx.x + (threadIdx.y + j + radius)*(blockDim.x)];
        float temp = d_kernel[(j + radius)*((radius << 1)+1)];
        value = d_uintToRGB(t);
        value *= temp;
        weight += temp;
        accumulation += value;
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}


/*
 *  Fast radius independent box filter
 *  Do Rows followed by Columns
 *  T = O(width + height)
 *  W = O(width*height + width*height)
 */
__global__ void d_boxFilterRow(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius)
{
    // memory location in d_img
    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x) * width;
    if (loc > height*width) return;

    d_img = d_img + loc;
    d_result = d_result + loc;
    int3 accumulation;
    int bWeight = (radius<<1) + 1; //all values in kernel weighted equally
    
    //initial clamping of left value
    accumulation = d_uintToRGB(d_img[0])*radius;
    for (int i = 0; i < radius + 1; i++) {
        accumulation += d_uintToRGB(d_img[i]); 
    }
    d_result[0] = d_rgbToUint(d_divide(accumulation, bWeight));

    for (int i = 1; i < radius + 1; i++) {
        accumulation += d_uintToRGB(d_img[i + radius]); 
        accumulation -= d_uintToRGB(d_img[0]); //clamp left side
        d_result[i] = d_rgbToUint(d_divide(accumulation, bWeight));
    }

    //resuses previous computed value
    for (int i = radius + 1; i < width - radius; i++) {
        accumulation += d_uintToRGB(d_img[i + radius]); 
        accumulation -= d_uintToRGB(d_img[i - radius - 1]); 
        d_result[i] = d_rgbToUint(d_divide(accumulation, bWeight));
    }

    for (int i = width - radius; i < width; i++){
        //clamp right side
        accumulation += d_uintToRGB(d_img[width - 1]); 
        accumulation -= d_uintToRGB(d_img[i - radius - 1]); 
        d_result[i] = d_rgbToUint(d_divide(accumulation, bWeight));
    }
}


/*
 *  Fast radius independent box filter
 *  Do Rows followed by Columns
 *  d_img should be d_result from the row filter
 *  T = O(width + height)
 *  W = O(width*height + width*height)
 */
__global__ void d_boxFilterCol(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius)
{
    // memory location in d_img
    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x);
    if (loc >= width) return;

    d_img = d_img + loc;
    d_result = d_result + loc;
    int3 accumulation;
    int bWeight = (radius<<1) + 1; //all values in kernel weighted equally
    

    //initial clamping of left value
    accumulation = d_uintToRGB(d_img[0])*radius;
    for (int i = 0; i < radius + 1; i++) {
        accumulation += d_uintToRGB(d_img[i * width]); 
    }
    d_result[0] = d_rgbToUint(d_divide(accumulation, bWeight));

    for (int i = 1; i < radius + 1; i++) {
        accumulation += d_uintToRGB(d_img[(i + radius) * width]); 
        accumulation -= d_uintToRGB(d_img[0]); //clamp left side
        d_result[i * width] = d_rgbToUint(d_divide(accumulation, bWeight));
    }

    //resuses previous computed value
    for (int i = radius + 1; i < height - radius; i++) {
        accumulation += d_uintToRGB(d_img[(i + radius)*width]); 
        accumulation -= d_uintToRGB(d_img[(i - radius)*width - width]); 
        d_result[i * width] = d_rgbToUint(d_divide(accumulation, bWeight));
    }

    for (int i = height - radius; i < height; i++){
        //clamp right side
        accumulation += d_uintToRGB(d_img[(height - 1)*width]); 
        accumulation -= d_uintToRGB(d_img[(i - radius)*width - width]); 
        d_result[i * width] = d_rgbToUint(d_divide(accumulation, bWeight));
    }
}

extern StopWatchInterface *timer;

/*
 * look at main.cpp kerboard interrupts for descriptions on what type and kernels do
 */
double convolution(unsigned int *d_img, unsigned int *d_result, int *h_kernel, int width, int height,
                 int radius, int type, int weight, int iterations)
{
    checkCudaErrors(cudaDeviceSynchronize());

    // threadsPerBlock needs to be a multiple of 32 for proper coalesce
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    //numBlocks should probably be a multiple of warp size here for proper coalesce..
    dim3 numBlocks(ceil((float)width / threadsPerBlock.x), ceil((float)height/threadsPerBlock.y));

    //copy kernel to device memory
    if (radius < 15)
        checkCudaErrors(cudaMemcpyToSymbol(d_kernel, h_kernel, ((radius << 1)+1)*((radius << 1)+1)*sizeof(int)));

    unsigned int *d_temp = NULL;
    if (type == 3)
        checkCudaErrors(cudaMalloc((void **) &d_temp, width*height*sizeof(unsigned int)));

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    for (int i = 0; i < iterations; i++) {
        switch (type) {
            case 0: 
                d_slowConvolution<<< numBlocks, threadsPerBlock>>>(d_img, d_result, width, height, radius, weight);
                break;
            case 1:
                d_sharedSlowConvolution<<< numBlocks, threadsPerBlock, (BLOCK_SIZE+(radius << 1))*(BLOCK_SIZE+(radius << 1))*sizeof(unsigned int)>>>(d_img, d_result, width, height, radius, weight);
                break;
            case 2:
                d_sepRowConvolution<<< numBlocks, threadsPerBlock, (BLOCK_SIZE+(radius << 1))*(BLOCK_SIZE)*sizeof(unsigned int)>>>(d_img, d_result, width, height, radius);
                d_sepColConvolution<<< numBlocks, threadsPerBlock, (BLOCK_SIZE)*(BLOCK_SIZE+(radius << 1))*sizeof(unsigned int)>>>(d_result, width, height, radius);
                break;
            case 3:
                d_boxFilterRow<<< ceil((float)height/BLOCK_SIZE), BLOCK_SIZE>>>(d_img, d_temp, width, height, radius);
                d_boxFilterCol<<< ceil((float)width/BLOCK_SIZE), BLOCK_SIZE>>>(d_temp, d_result, width, height, radius);
                break;
        }
        checkCudaErrors(cudaDeviceSynchronize());
        d_img = d_result;
    }
    sdkStopTimer(&timer);
    printf("time taken: %f\n", sdkGetTimerValue(&timer));

    checkCudaErrors(cudaFree(d_temp));
    return 0;
}