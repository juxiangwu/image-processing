#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"

__global__ void GPU_histogram_kernal(int * hist_out, unsigned char * img_in, int nbr_bin, int img_size)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int shared_hist[256];

	shared_hist[threadIdx.x] = 0;

	__syncthreads();

	if (i < img_size)
		atomicAdd(&shared_hist[img_in[i]], 1);

	__syncthreads();

	/* Copy to Global */
	atomicAdd(&hist_out[threadIdx.x], shared_hist[threadIdx.x]); 
}

void GPU_histogram(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin)
{
  	int *gpu_hist_out;
    unsigned char *gpu_img_in;

    for (int i = 0; i < nbr_bin; i++)
    	hist_out[i] = 0;

 	cudaMalloc(&gpu_hist_out, nbr_bin * sizeof(int));
    cudaMalloc(&gpu_img_in, sizeof(unsigned char) * img_size);

    cudaMemcpy(gpu_img_in, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_hist_out, hist_out, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice);

	dim3 blockDim(nbr_bin);
	dim3 gridDim((img_size + blockDim.x - 1)/ blockDim.x);

    GPU_histogram_kernal<<<gridDim, blockDim>>>(gpu_hist_out, gpu_img_in, nbr_bin, img_size);

    cudaMemcpy(hist_out, gpu_hist_out, sizeof(int) * nbr_bin, cudaMemcpyDeviceToHost);   

    cudaFree(gpu_hist_out);
    cudaFree(gpu_img_in);
}

__global__ void GPU_histogram_equalization_kernal(int *cdf_table, unsigned char *img_out, unsigned char * img_in, int img_size, int nbr_bin, int min, int d)
{
	__shared__ int lut[256];

	int i = threadIdx.x;
    if (i < nbr_bin)
    {
        lut[i] = (int)(((float)cdf_table[i] - min)*255/d + 0.5);
        if (lut[i] < 0)
         {
            lut[i] = 0;
        }     
    }
    __syncthreads();

    i = blockIdx.x * blockDim.x + threadIdx.x;
    img_out[i] = lut[img_in[i]];
}

void GPU_histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin)
{
    int cdf_table[256];
    int i, cdf, min, d;

    int *gpu_cdf_table;
    unsigned char *gpu_img_in;
  	unsigned char *gpu_img_out;

    cdf = 0;
    min = 0;
    i = 0;

    /* Calculate the min, best done serially. */
    while(min == 0)
    {
        min = hist_in[i++];
    }

    d = img_size - min;

    /* Construct the CDF Table, best done serially. */
    for(i = 0; i < nbr_bin; i++)
    {
        cdf += hist_in[i];
        cdf_table[i] = cdf;
    }

    // Copy to GPU memory

    cudaMalloc(&gpu_cdf_table, sizeof(int) * nbr_bin);
    cudaMalloc(&gpu_img_in, sizeof(unsigned char) * img_size);
    cudaMalloc(&gpu_img_out, sizeof(unsigned char) * img_size);

    cudaMemcpy(gpu_cdf_table, cdf_table, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_img_in, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

	dim3 blockDim(nbr_bin);
	dim3 gridDim((img_size + blockDim.x - 1)/ blockDim.x);

    GPU_histogram_equalization_kernal<<<gridDim, blockDim>>>(gpu_cdf_table, gpu_img_out, gpu_img_in, nbr_bin, img_size, min, d);

    /* Copy image out */
    cudaMemcpy(img_out, gpu_img_out, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);  

    cudaFree(gpu_cdf_table);
    cudaFree(gpu_img_in);
    cudaFree(gpu_img_out);
}