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

#include <stdio.h>
#include <stdlib.h>

#include "RTGPUDefs.h"

texture<unsigned char, 2>	g_tex1Gray;				// the gray texture
texture<uchar4, 2>			g_tex1;					// the rgb texture
texture<float, 2>			g_tex1Float;			// the float intermediate
texture<float, 2>			g_kernel;				// the kernel texture

__constant__ float			g_sepKernel[MAX_GAUSSIAN_KERNEL_LENGTH];			// separable kernel

cudaArray	*kernel = NULL;

__global__ void kernelConvTexGrayRow(float *inter, int w, int rad)
{ 
    float	*intPtr = inter + blockIdx.x * w;
    float	res;
     
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) 
    {
		res = 0;
		#pragma unroll
		for (int x = -rad; x <= rad; x++) {
			res += (float)tex2D(g_tex1Gray, (float)i + x, (float)blockIdx.x) * g_sepKernel[x+rad];
		}
		intPtr[i] = res;
    }
}

__global__ void kernelConvTexGrayCol(unsigned char *output, int w, int rad)
{ 
    unsigned char	*out = output + blockIdx.x * w;
    float	sum;
    uchar4	res;
     
    for ( int i = 4 * threadIdx.x; i < w; i += 4 * blockDim.x ) 
    {
		sum = 0;
		#pragma unroll
		for (int y = -rad; y <= rad; y++) {
			sum += tex2D(g_tex1Float, (float)i, (float)blockIdx.x + y) * g_sepKernel[y+rad];
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.x = sum;

		sum = 0;
		#pragma unroll
		for (int y = -rad; y <= rad; y++) {
			sum += tex2D(g_tex1Float, (float)(i+1), (float)blockIdx.x + y) * g_sepKernel[y+rad];
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.y = sum;

		sum = 0;
		#pragma unroll
		for (int y = -rad; y <= rad; y++)
		{
			sum += tex2D(g_tex1Float, (float)(i+2), (float)blockIdx.x + y) * g_sepKernel[y+rad];
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.z = sum;

		sum = 0;
		#pragma unroll
		for (int y = -rad; y <= rad; y++)
		{
			sum += tex2D(g_tex1Float, (float)(i+3), (float)blockIdx.x + y) * g_sepKernel[y+rad];
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.w = sum;

		*(uchar4 *)(out + i) = res;
    }
}


__global__ void kernelConvTexGray( unsigned char *output, int w, int rad)
{ 
    unsigned char *out = output + blockIdx.x * w;
    float	sum;
    uchar4	res;
     
    for ( int i = 4 * threadIdx.x; i < w; i += 4 * blockDim.x ) 
    {
		sum = 0;
		for (int y = -rad; y <= rad; y++)
		{
			#pragma unroll
			for (int x = -rad; x <= rad; x++)
			{
				sum += (float)tex2D(g_tex1Gray, (float)i + x, (float)blockIdx.x + y) * tex2D(g_kernel, (float)(x+rad), (float)(y + rad));
			}
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.x = sum;
		
		sum = 0;
		for (int y = -rad; y <= rad; y++)
		{
			#pragma unroll
			for (int x = -rad; x <= rad; x++)
			{
				sum += (float)tex2D(g_tex1Gray, (float)i + x + 1, (float)blockIdx.x + y) * tex2D(g_kernel, (float)(x+rad), (float)(y + rad));
			}
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.y = sum;
		
		sum = 0;
		for (int y = -rad; y <= rad; y++)
		{
			#pragma unroll
			for (int x = -rad; x <= rad; x++)
			{
				sum += (float)tex2D(g_tex1Gray, (float)i + x + 2, (float)blockIdx.x + y) * tex2D(g_kernel, (float)(x+rad), (float)(y + rad));
			}
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.z = sum;
		
		sum = 0;
		for (int y = -rad; y <= rad; y++)
		{
			#pragma unroll
			for (int x = -rad; x <= rad; x++)
			{
				sum += (float)tex2D(g_tex1Gray, (float)i + x + 3, (float)blockIdx.x + y) * tex2D(g_kernel, (float)(x+rad), (float)(y + rad));
			}
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.w = sum;
		*(uchar4 *)(out + i) = res;
    }
}


__global__ void kernelConvTex( uchar4 *output, int w, int rad)
{ 
    uchar4 *out = output + blockIdx.x * w;
    float sum;
    uchar4 res;
    
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) 
    {
		sum = 0;
		for (int y = -rad; y <= rad; y++)
		{
			for (int x = -rad; x <= rad; x++)
			{
				sum += (float)tex2D(g_tex1, (float)i + x, (float)blockIdx.x + y).x * tex2D(g_kernel, (float)(x+rad), (float)(y + rad));
			}
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.x = sum;
			
		sum = 0;
		for (int y = -rad; y <= rad; y++) {
			for (int x = -rad; x <= rad; x++)	{
					sum += (float)tex2D(g_tex1, (float)i + x, (float)blockIdx.x + y).y * tex2D(g_kernel, (float)(x+rad), (float)(y + rad));
			}
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.y = sum;

		sum = 0;
		for (int y = -rad; y <= rad; y++) {
			for (int x = -rad; x <= rad; x++) {
				sum += (float)tex2D(g_tex1, (float)i + x, (float)blockIdx.x + y).z * tex2D(g_kernel, (float)(x+rad), (float)(y + rad));
			}
		}
		if (sum > (float)255.0)
			sum = 255;
		if (sum < 0)
			sum = 0;
		res.z = sum;
	
		out[i] = res;
    }
}

extern "C" int _RTGPUCreateConvTex(int rad) 
{
	int i, n, twoexpn, fact, f1, f2, len, x, y;
	char msg[1000];
	float k[MAX_GAUSSIAN_KERNEL_LENGTH];
	float k2[MAX_GAUSSIAN_KERNEL_LENGTH * MAX_GAUSSIAN_KERNEL_LENGTH];
    cudaChannelFormatDesc desc;

	RTGPUTrace("RTGPUCreateConvTex");
	if (rad > MAX_GAUSSIAN_KERNEL_RADIUS)
	{
		RTGPUTrace("Incorrect kernel size");
		return 0;
	}

	len = 2 * rad + 1;

	twoexpn = 1;
	fact = 1;
	for (n = 0; n < (2 * rad); n+=2)
	{

		fact *= (n+1)*(n+2);
		twoexpn *= 4;
	}	
	sprintf(msg, "nfact = %d, twoexpn = %d", fact, twoexpn);
	RTGPUTrace(msg);
	f1 = 1;
	f2 = fact;
	for (i = 0; i <= n; i++)
	{
		k[i] = (float)(fact/(f1 * f2))/(float)twoexpn;
		if (i == n)
			break;
		f1 *= (i+1);
		f2 /= n-i;
	}	
	RTGPUSafeCall(cudaMemcpyToSymbol(g_sepKernel, k, MAX_GAUSSIAN_KERNEL_LENGTH * sizeof(float)));

//	Now compute square array

	for (y = 0; y < len; y++) {
		for (x = 0; x < len; x++) {
			k2[y*len + x] = k[x] * k[y];
			sprintf(msg, "  x = %d, y = %d, k2 = %f", x, y, k2[y * len + x]);
			RTGPUTrace(msg);
		}
	}

	if (kernel != NULL)
		RTGPUSafeCall(cudaFreeArray(kernel));
	desc = cudaCreateChannelDesc<float>();
	RTGPUSafeCall(cudaMallocArray(&kernel, &desc, len, len));
	RTGPUSafeCall(cudaMemcpyToArray(kernel, 0, 0, k2, len * len * sizeof(float), cudaMemcpyHostToDevice));
	
    return 1;
}


extern "C" int _RTGPUConvTex(int srcSlot, int destSlot, int rad) 
{
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);
	
	RTGPUTrace("RTGPUConvTex");

	RTGPUSafeCall(cudaBindTextureToArray(g_kernel, kernel));

	_RTGPUSetupSlot(DI, SI->width, SI->height, SI->color);
	
	if (!SI->color) {
		desc = cudaCreateChannelDesc<unsigned char>();
		RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex1Gray, SI->image, desc, SI->width, SI->height, SI->width));

		kernelConvTexGrayRow<<<SI->height, 256>>>((float *)SI->inter, SI->width, rad);

		RTGPUSafeCall(cudaUnbindTexture(g_tex1Gray));

		desc = cudaCreateChannelDesc<float>();
		RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex1Float, SI->inter, desc, SI->width, SI->height, SI->width * 4));

		kernelConvTexGrayCol<<<SI->height, 256>>>((unsigned char *)DI->image, SI->width, rad);

		RTGPUSafeCall(cudaUnbindTexture(g_tex1Float));
	} else {
		desc = cudaCreateChannelDesc<uchar4>();
		RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex1, SI->image, desc, SI->width, SI->height, SI->width * 4));

		kernelConvTex<<<SI->height, 256>>>(DI->image, SI->width, rad);

		RTGPUSafeCall(cudaUnbindTexture(g_tex1));
	}
	RTGPUSafeCall(cudaUnbindTexture(g_kernel));
		
    return 1;
}