//
//  Copyright (c) 2014 richards-tech
//
//  This file is part of RTGPULib
//
//  RTGPULib is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  RTGPULib is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with RTGPULib.  If not, see <http://www.gnu.org/licenses/>.
//
//
//	This software contains source code provided by NVIDIA Corporation.

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#include "RTGPUDefs.h"

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[MAX_GAUSSIAN_KERNEL_LENGTH];
static int	gnKRad;				// kernel radius
static int	gnKLen;				// kernel length


extern "C" void _RTGPUSetGaussianKernel(float *h_Kernel, int rad)
{
	gnKRad = rad;
	gnKLen = rad * 2 + 1;
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, gnKLen * sizeof(float));
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 8
#define	  ROWS_RESULT_STEPS 4
#define   ROWS_HALO_STEPS 1
#define	  ROWS_SLICELEN	((ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X)

__global__ void kernelConvolutionRowsKernelGray(unsigned char *src, float *inter, int imageW, int imageH, int rad)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][ROWS_SLICELEN];
     float	fsum;

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    src += baseY * imageW + baseX;
    inter += baseY * imageW + baseX;

    //Load main data
    #pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = src[i * ROWS_BLOCKDIM_X];
	}

	//Load left halo
	#pragma unroll
	for(int i = 0; i < ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (float)((baseX >= -i * ROWS_BLOCKDIM_X ) ? src[i * ROWS_BLOCKDIM_X] : 0);
	}
	//Load right halo
	#pragma unroll
    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 0 * ROWS_SLICELEN] = (float)((imageW - baseX > i * ROWS_BLOCKDIM_X) ? src[i * ROWS_BLOCKDIM_X] : 0);
	}
	 
    //Compute and store results
    __syncthreads();
    
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
		
		fsum = 0;
        #pragma unroll
        for(int j = -rad; j <= rad; j++)
            fsum += c_Kernel[rad - j] * s_Data[threadIdx.y][threadIdx.x + j + i * ROWS_BLOCKDIM_X];
            
        inter[i * ROWS_BLOCKDIM_X] = fsum;
   }
}

__global__ void kernelConvolutionRowsKernel(uchar4 *src, float *inter, int imageW, int imageH, int rad)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][ROWS_SLICELEN * 4];
     float4	fsum;
     uchar4 val;

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    src += baseY * imageW + baseX;
    inter += 4 * baseY * imageW + baseX;

    //Load main data
    #pragma unroll
	for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
		val = src[i * ROWS_BLOCKDIM_X];
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 0 * ROWS_SLICELEN] = (float)val.x;
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 1 * ROWS_SLICELEN] = (float)val.y;
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 2 * ROWS_SLICELEN] = (float)val.z;
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 3 * ROWS_SLICELEN] = (float)val.w;
	}

	//Load left halo
	#pragma unroll
	for(int i = 0; i < ROWS_HALO_STEPS; i++) {
		val = src[i * ROWS_BLOCKDIM_X];
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 0 * ROWS_SLICELEN] = (float)((baseX >= -i * ROWS_BLOCKDIM_X ) ? val.x : 0);
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 1 * ROWS_SLICELEN] = (float)((baseX >= -i * ROWS_BLOCKDIM_X ) ? val.y : 0);
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 2 * ROWS_SLICELEN] = (float)((baseX >= -i * ROWS_BLOCKDIM_X ) ? val.z : 0);
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 3 * ROWS_SLICELEN] = (float)((baseX >= -i * ROWS_BLOCKDIM_X ) ? val.w : 0);
	}
	//Load right halo
	#pragma unroll
    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
		val = src[i * ROWS_BLOCKDIM_X];
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 0 * ROWS_SLICELEN] = (float)((imageW - baseX > i * ROWS_BLOCKDIM_X) ? val.x : 0);
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 1 * ROWS_SLICELEN] = (float)((imageW - baseX > i * ROWS_BLOCKDIM_X) ? val.y : 0);
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 2 * ROWS_SLICELEN] = (float)((imageW - baseX > i * ROWS_BLOCKDIM_X) ? val.z : 0);
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + 3 * ROWS_SLICELEN] = (float)((imageW - baseX > i * ROWS_BLOCKDIM_X) ? val.w : 0);
	}
	 
    //Compute and store results
    __syncthreads();
    
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
		
		fsum.x = 0;
        #pragma unroll
        for(int j = -rad; j <= rad; j++)
            fsum.x += c_Kernel[rad - j] * s_Data[threadIdx.y][threadIdx.x + j + i * ROWS_BLOCKDIM_X + 0 * ROWS_SLICELEN];
            
 		fsum.y = 0;
	    #pragma unroll
		for(int j = -rad; j <= rad; j++)
			fsum.y += c_Kernel[rad - j] * s_Data[threadIdx.y][threadIdx.x + j + i * ROWS_BLOCKDIM_X + 1 * ROWS_SLICELEN];
 	
		fsum.z = 0;
		#pragma unroll
		for(int j = -rad; j <= rad; j++)
			fsum.z += c_Kernel[rad - j] * s_Data[threadIdx.y][threadIdx.x + j + i * ROWS_BLOCKDIM_X + 2 * ROWS_SLICELEN];
        
		fsum.w = 0;
		#pragma unroll
		for(int j = -rad; j <= rad; j++)
			fsum.w += c_Kernel[rad - j] * s_Data[threadIdx.y][threadIdx.x + j + i * ROWS_BLOCKDIM_X + 3 * ROWS_SLICELEN];
		
        inter[i * ROWS_BLOCKDIM_X + 0 * imageW] = fsum.x;
		inter[i * ROWS_BLOCKDIM_X + 1 * imageW] = fsum.y;
		inter[i * ROWS_BLOCKDIM_X + 2 * imageW] = fsum.z;
		inter[i * ROWS_BLOCKDIM_X + 3 * imageW] = fsum.w;
    }
}

extern "C" void _RTGPUConvolutionRowsGPU(uchar4 *src, float *inter, int imageW, int imageH, bool color)
{
 	RTGPUTrace("RTConvolutionRowsGPU");
    assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= gnKRad );
    assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert( imageH % ROWS_BLOCKDIM_Y == 0 );

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	if (!color)
		kernelConvolutionRowsKernelGray<<<blocks, threads>>>((unsigned char *)src, inter, imageW, imageH, gnKRad);
	else
		kernelConvolutionRowsKernel<<<blocks, threads>>>(src, inter, imageW, imageH, gnKRad);
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 1
#define		COLUMNS_SLICELEN	((COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1)

__global__ void kernelConvolutionColumnsKernelGray(float *inter, unsigned char *dst, int imageW, int imageH, int rad)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][COLUMNS_SLICELEN];
	float	sum;
	
    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    inter += baseY * imageW + baseX;
    dst += baseY * imageW + baseX;

    //Main data
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = inter[i * COLUMNS_BLOCKDIM_Y * imageW];
    }

    //Upper halo
    #pragma unroll
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++) {
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? inter[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
	}
    //Lower halo
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? inter[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
	}
    //Compute and store results
    __syncthreads();
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        sum = 0;
        #pragma unroll
        for(int j = -rad; j <= rad; j++)
        {
            sum += c_Kernel[rad - j] * s_Data[threadIdx.x][threadIdx.y + j + i * COLUMNS_BLOCKDIM_Y];
        }
		if (sum > 255)
			sum = 255;
		if (sum < 0)
			sum = 0;
        dst[i * COLUMNS_BLOCKDIM_Y * imageW] = (unsigned char)sum;
   }
}


__global__ void kernelConvolutionColumnsKernel(float *inter, uchar4 *dst, int imageW, int imageH, int rad)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][COLUMNS_SLICELEN * 4];
	uchar4	cres;
	float	sum;
	
    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    inter += 4 * baseY * imageW + baseX;
    dst += baseY * imageW + baseX;

    //Main data
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 0 * COLUMNS_SLICELEN] = inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 0];
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 1 * COLUMNS_SLICELEN] = inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 1];
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 2 * COLUMNS_SLICELEN] = inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 2];
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 3 * COLUMNS_SLICELEN] = inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 3];
	}

    //Upper halo
    #pragma unroll
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++) {
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 0 * COLUMNS_SLICELEN] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 0] : 0;
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 1 * COLUMNS_SLICELEN] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 1] : 0;
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 2 * COLUMNS_SLICELEN] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 2] : 0;
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 3 * COLUMNS_SLICELEN] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 3] : 0;
	}

    //Lower halo
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 0 * COLUMNS_SLICELEN]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 0] : 0;
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 1 * COLUMNS_SLICELEN]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 1] : 0;
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 2 * COLUMNS_SLICELEN]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 2] : 0;
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + 3 * COLUMNS_SLICELEN]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? inter[4 * i * COLUMNS_BLOCKDIM_Y * imageW + imageW * 3] : 0;
	}
    //Compute and store results
    __syncthreads();
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        sum = 0;
        #pragma unroll
        for(int j = -rad; j <= rad; j++)
            sum += c_Kernel[rad - j] * s_Data[threadIdx.x][threadIdx.y + j + i * COLUMNS_BLOCKDIM_Y + 0 * COLUMNS_SLICELEN];
		if (sum > 255)
			sum = 255;
		if (sum < 0)
			sum = 0;
		cres.x = (unsigned char)sum;
 
		sum = 0;
		#pragma unroll
		for(int j = -rad; j <= rad; j++)
			sum += c_Kernel[rad - j] * s_Data[threadIdx.x][threadIdx.y + j + i * COLUMNS_BLOCKDIM_Y + 1 * COLUMNS_SLICELEN];
		if (sum > 255)
			sum = 255;
		if (sum < 0)
			sum = 0;
		cres.y = (unsigned char)sum;  
  
		sum = 0;
		#pragma unroll
		for(int j = -rad; j <= rad; j++)
			sum += c_Kernel[rad - j] * s_Data[threadIdx.x][threadIdx.y + j + i * COLUMNS_BLOCKDIM_Y + 2 * COLUMNS_SLICELEN];
		if (sum > 255)
			sum = 255;
		if (sum < 0)
			sum = 0;
		cres.z = (unsigned char)sum;  

        dst[i * COLUMNS_BLOCKDIM_Y * imageW] = cres;
   }
}

extern "C" void _RTGPUConvolutionColumnsGPU(float *inter, uchar4 *dst, int imageW, int imageH, bool color)
{
    assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= gnKRad );
    assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

 	RTGPUTrace("RTGPUConvolutionColumnsGPU");
    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
	
	if (!color)
		kernelConvolutionColumnsKernelGray<<<blocks, threads>>>(inter, (unsigned char *)dst, imageW, imageH, gnKRad);
	else
		kernelConvolutionColumnsKernel<<<blocks, threads>>>(inter, dst, imageW, imageH, gnKRad);
}


extern "C" int _RTGPUSetGaussian(int rad)
{
	int i, n, twoexpn, fact, f1, f2;
	char msg[1000];
	float kernel[MAX_GAUSSIAN_KERNEL_LENGTH];

	RTGPUTrace("RTGPUSetGaussian");
	if (rad > MAX_GAUSSIAN_KERNEL_RADIUS) {
		RTGPUTrace("Incorrect kernel size");
		return 0;
	}

	twoexpn = 1;
	fact = 1;
	for (n = 0; n < (2 * rad); n+=2) {

		fact *= (n+1)*(n+2);
		twoexpn *= 4;
	}	
	sprintf(msg, "fact = %d, twoexpn = %d", fact, twoexpn);
	RTGPUTrace(msg);
	f1 = 1;
	f2 = fact;
	for (i = 0; i <= n; i++) {
		kernel[i] = (float)(fact/(f1 * f2))/(float)twoexpn;
		sprintf(msg, "  i = %d, f1 = %d, f2 = %d, pK = %f", i, f1, f2, kernel[i]);
		RTGPUTrace(msg);
		if (i == n)
			break;
		f1 *= (i+1);
		f2 /= n-i;
	}	
	_RTGPUSetGaussianKernel(kernel, rad);
	return 1;
}

extern "C" int _RTGPUGaussian(int srcSlot, int destSlot, int rad) 
{
	RTGPU_IMAGE *SI, *DI;
	
	RTGPUTrace("RTGPUGaussian");
	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);
	_RTGPUSetupSlot(DI, SI->width, SI->height, SI->color);

	if (rad == 0) {						// just copy image!
		RTGPUSafeCall(cudaMemcpy(DI->image, SI->image, SI->width * SI->height * sizeof(uchar4), cudaMemcpyDeviceToDevice));
		return 1;
	}
	
	_RTGPUConvolutionRowsGPU(SI->image, (float *)SI->inter, SI->width, SI->height, SI->color);
	_RTGPUConvolutionColumnsGPU((float *)SI->inter, DI->image, SI->width, SI->height, SI->color);
	
	return 1;
}
