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

texture<unsigned char, 2, cudaReadModeNormalizedFloat> boxFilterTexGray;
texture<uchar4, 2, cudaReadModeNormalizedFloat> boxFilterTex;

cudaArray* d_array, *d_tempArray;

/*
    Perform a fast box filter using the sliding window method.
    As the kernel moves from left to right, we add in the contribution of the new
    sample on the right, and subtract the value of the exiting sample on the left.
    This only requires 2 adds and a mul per output value, independent of the filter radius.
    The box filter is separable, so to perform a 2D box filter we perform the filter in
    the x direction, followed by the same filter in the y direction.
    Applying multiple iterations of the box filter converges towards a Gaussian blur.
    Using CUDA, rows or columns of the image are processed in parallel.
    This version duplicates edge pixels.
    Parameters
    id - pointer to input data in global memory
    od - pointer to output data in global memory
    w  - image width
    h  - image height
    r  - filter radius
    e.g. for r = 2, w = 8:
    0 1 2 3 4 5 6 7
    x - -
    - x - -
    - - x - -
      - - x - -
        - - x - -
          - - x - -
            - - x -
              - - x
*/

__device__ uint grayFloatToInt(float val)
{
    return (uint(__saturatef(val) * 255.0f));
}


__device__ uint rgbFloatToInt(float4 rgb)
{
    rgb.x = __saturatef(rgb.x);   // clamp to [0.0, 1.0]
    rgb.y = __saturatef(rgb.y);
    rgb.z = __saturatef(rgb.z);
    return (uint(rgb.z * 255.0f) << 16) | (uint(rgb.y * 255.0f) << 8) | uint(rgb.x * 255.0f);
}

__device__ float grayIntToFloat(uint c)
{
    float gray;
    gray = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    return gray;
}

__device__ float4 rgbIntToFloat(uint c)
{
    float4 rgb;
    rgb.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgb.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgb.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgb;
}

// row pass using texture lookups

__global__ void kernelBoxFilterGrayRow(unsigned char *od, int w, int h, int r, float scale)
{
 
	unsigned int y = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // as long as address is always less than height, we do work
    if (y < h) 
    {
        float t = 0;
        for(int x = -r; x <= r; x++) 
        {
            t += tex2D(boxFilterTexGray, x, y);
        }
        od[y * w] = (unsigned char)(__saturatef(t * scale) * 255.0f);

        for(int x = 1; x < w; x++) 
        {
            t += tex2D(boxFilterTexGray, x + r, y);
            t -= tex2D(boxFilterTexGray, x - r - 1, y);
            od[y * w + x] = (unsigned char)(__saturatef(t * scale) * 255.0f);
        }
    }
}

__global__ void kernelBoxFilterRGBRow(uint *od, int w, int h, int r, float scale)
{
	unsigned int y = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // as long as address is always less than height, we do work
    if (y < h) 
    {
        float4 t = make_float4(0.0f);
        for(int x = -r; x <= r; x++) 
        {
            t += tex2D(boxFilterTex, x, y);
        }
        od[y * w] = rgbFloatToInt(t * scale);

        for(int x = 1; x < w; x++) 
        {
            t += tex2D(boxFilterTex, x + r, y);
            t -= tex2D(boxFilterTex, x - r - 1, y);
            od[y * w + x] = rgbFloatToInt(t * scale);
        }
    }
}
// column pass using texture memory

__global__ void kernelBoxFilterGrayCol(unsigned char *od, int w, int h, int r, float scale)
{
	unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (x < w)
	{
		float t = 0.0f;
		for(int y = -r; y <= r; y++) 
		{
			t += tex2D(boxFilterTexGray, x, y);
		}
		od[x] = (unsigned char)(__saturatef(t * scale) * 255.0f);

		for(int y = 1; y < h; y++) 
		{
			t += tex2D(boxFilterTexGray, x, y + r);
			t -= tex2D(boxFilterTexGray, x, y - r - 1);
			od[y * w + x] = (unsigned char)(__saturatef(t * scale) * 255.0f);
		}
	}
}

//	RGB uses coalesced global memory reads instead of texture

__global__ void kernelBoxFilterRGBCol(uint *id, uint *od, int w, int h, int r, float scale)
{
	unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (x >= w)
		return;
    id = &id[x];
    od = &od[x];

    float4 t;
    // do left edge
    t = rgbIntToFloat(id[0]) * r;
    for (int y = 0; y < (r + 1); y++) 
    {
        t += rgbIntToFloat(id[y*w]);
    }
    od[0] = rgbFloatToInt(t * scale);

    for(int y = 1; y < (r + 1); y++) 
    {
        t += rgbIntToFloat(id[(y + r) * w]);
        t -= rgbIntToFloat(id[0]);
        od[y * w] = rgbFloatToInt(t * scale);
    }
    
    // main loop
    for(int y = (r + 1); y < (h - r); y++) 
    {
        t += rgbIntToFloat(id[(y + r) * w]);
        t -= rgbIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbFloatToInt(t * scale);
    }

    // do right edge
    for (int y = h - r; y < h; y++) 
    {
        t += rgbIntToFloat(id[(h - 1) * w]);
        t -= rgbIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbFloatToInt(t * scale);
    }
}

extern "C" int _RTGPUBoxFilter(int srcSlot, int destSlot, int rad, bool normalize)
{
	int w, h;
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;
	int			blocksH, blocksW, threads;
	float		scale;
	
	RTGPUTrace("RTGPUBoxFilter");

	assert(sizeof(unsigned int) == sizeof(uchar4));
	
	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	w = SI->width;
	h = SI->height;

	_RTGPUSetupSlot(DI, w, h, SI->color);
	
	if (rad == 0)
	{						// just copy image!
		if (!SI->color)
			checkCudaErrors(cudaMemcpy(DI->image, DI->image, w * h, cudaMemcpyDeviceToDevice));
		else
			checkCudaErrors(cudaMemcpy(DI->image, SI->image, w * h * 4, cudaMemcpyDeviceToDevice));
		return 1;
	}

	scale = normalize ? 1.0f / (float)((rad << 1) + 1) : 1.0f; 

	threads = 16;
	blocksH = (DI->height + threads - 1) / threads;
	blocksW = DI->width / threads;

	if (!SI->color)	{
		desc = cudaCreateChannelDesc<unsigned char>();
		RTGPUSafeCall(cudaBindTexture2D(NULL, boxFilterTexGray, SI->image, desc, w, h, w));

		kernelBoxFilterGrayRow<<< blocksH, threads, 0 >>>((unsigned char *)SI->inter, w, h, rad, scale);

		RTGPUSafeCall(cudaBindTexture2D(NULL, boxFilterTexGray, SI->inter, desc, w, h, w));

		kernelBoxFilterGrayCol<<< blocksW, threads, 0 >>>((unsigned char *)DI->image, w, h, rad, scale);

		RTGPUSafeCall(cudaUnbindTexture(boxFilterTexGray));
	} else {
		desc = cudaCreateChannelDesc<uchar4>();
		RTGPUSafeCall(cudaBindTexture2D(NULL, boxFilterTex, SI->image, desc, w, h, w * 4));

		kernelBoxFilterRGBRow<<< blocksH, threads, 0 >>>((uint *)SI->inter, w, h, rad, scale);
		kernelBoxFilterRGBCol<<< blocksW, threads, 0 >>>((uint *)SI->inter, (unsigned int *)DI->image, 
				w, h, rad, scale);

		RTGPUSafeCall(cudaUnbindTexture(boxFilterTex));
	}

	return 1;
}