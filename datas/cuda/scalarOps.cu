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
//
//	Some code is derived from OpenCV - original License Agreement is shown below

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "RTGPUDefs.h"

__constant__ unsigned char	gpLUT[256];				// lookup table for GPU_SCALAROPS_LUT

//	kernelScalarOpsAdd is a saturating scalar add function. Returns nA + nB.
//

__device__ unsigned char kernelScalarOpsAdd(unsigned char nA, unsigned char nB)
{
	unsigned int val;

	val = (unsigned int)nA + (unsigned int)nB;
	if (val > 255)
		val = 255;
	return (unsigned char)val;
}

//	kernelScalarOpsAddMask is a saturating scalar add function. *nRes = nA + nB.
//

__device__ void kernelScalarOpsAddMask(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	unsigned int val;

	if (nMask == 0)
		return;
	val = (unsigned int)nA + (unsigned int)nB;
	if (val > 255)
		val = 255;
	*pnRes = (unsigned char)val;
}


//	kernelScalarOpsSub is a saturating scalar subtract function. Returns nA - nB.
//

__device__ unsigned char kernelScalarOpsSub(unsigned char nA, unsigned char nB)
{
	int val;

	val = (int)nA - (int)nB;
	if (val < 0)
		val = 0;
	return (unsigned char)val;
}


//	kernelScalarOpsSubMask is a saturating scalar subtract function. *pnRes = nA - nB.
//

__device__ void kernelScalarOpsSubMask(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	int val;

	if (nMask == 0)
		return;
	val = (int)nA - (int)nB;
	if (val < 0)
		val = 0;
	*pnRes = (unsigned char)val;
}

__global__ void kernelScalarOpsGray(uchar4 *pInput, uchar4 *pOutput, float v0, int nW, int nType)
{ 
	uchar4		*pIn = pInput + blockIdx.x * nW / 4;
	uchar4		*pOut = pOutput + blockIdx.x * nW / 4;
    uchar4		src;
    uchar4		res;
	int			i, j;
     
    for (i = threadIdx.x, j = 4 * threadIdx.x; j < nW; i += blockDim.x, j += 4 * blockDim.x ) 
    {
        src = pIn[i];
	    switch (nType)
	    {
			case GPU_SCALAROPS_SUB:
				res.x = kernelScalarOpsSub(src.x, v0);
				res.y = kernelScalarOpsSub(src.y, v0);
				res.z = kernelScalarOpsSub(src.z, v0);
				res.w = kernelScalarOpsSub(src.w, v0);
				break;

			case GPU_SCALAROPS_ADD:
				res.x = kernelScalarOpsAdd(src.x, v0);
				res.y = kernelScalarOpsAdd(src.y, v0);
				res.z = kernelScalarOpsAdd(src.z, v0);
				res.w = kernelScalarOpsAdd(src.w, v0);
				break;

			case GPU_SCALAROPS_NOT:
				res.x = ~src.x;
				res.y = ~src.y;
				res.z = ~src.z;
				res.w = ~src.w;
				break;

			case GPU_SCALAROPS_LUT:
				res.x = gpLUT[src.x];
				res.y = gpLUT[src.y];
				res.z = gpLUT[src.z];
				res.w = gpLUT[src.w];
				break;
		}
 		pOut[i] = res;
    }
}


__global__ void kernelScalarOpsGrayMask(uchar4 *pInput, uchar4 *pOutput, uchar4 *pMask, float v0, int nW, int nType)
{ 
	uchar4		*pIn = pInput + blockIdx.x * nW / 4;
	uchar4		*pM = pMask + blockIdx.x * nW / 4;
	uchar4		*pOut = pOutput + blockIdx.x * nW / 4;
    uchar4		src;
    uchar4		res;
	uchar4		mask;
	int			i, j;
     
    for (i = threadIdx.x, j = 4 * threadIdx.x; j < nW; i += blockDim.x, j += 4 * blockDim.x ) 
    {
        src = pIn[i];
		mask = pM[i];
		res = pOut[i];

		switch (nType)
	    {
			case GPU_SCALAROPS_SUB:
				kernelScalarOpsSubMask(src.x, v0, &(res.x), mask.x);
				kernelScalarOpsSubMask(src.y, v0, &(res.y), mask.y);
				kernelScalarOpsSubMask(src.z, v0, &(res.z), mask.z);
				kernelScalarOpsSubMask(src.w, v0, &(res.w), mask.w);
				break;

			case GPU_SCALAROPS_ADD:
				kernelScalarOpsAddMask(src.x, v0, &(res.x), mask.x);
				kernelScalarOpsAddMask(src.y, v0, &(res.y), mask.y);
				kernelScalarOpsAddMask(src.z, v0, &(res.z), mask.z);
				kernelScalarOpsAddMask(src.w, v0, &(res.w), mask.w);
				break;

			case GPU_SCALAROPS_NOT:
				if (mask.x != 0)
					res.x = ~src.x;
				else
					res.x = src.x;
				if (mask.y != 0)
					res.y = ~src.y;
				else
					res.y = src.y;
				if (mask.z != 0)
					res.z = ~src.z;
				else
					res.z = src.z;
				break;

			case GPU_SCALAROPS_LUT:
				res.x = gpLUT[src.x];
				res.y = gpLUT[src.y];
				res.z = gpLUT[src.z];
				res.w = gpLUT[src.w];
				break;
		}
 		pOut[i] = res;
    }
}

__global__ void kernelScalarOpsRGB(uchar4 *pInput, uchar4 *pOutput, float v0, float v1, float v2, float v3, int nW, int nType)
{ 
    uchar4		src;
    uchar4		res;
 	uchar4		*pIn = pInput + blockIdx.x * nW;
	uchar4		*pOut = pOutput + blockIdx.x * nW;
 
    for ( int i = threadIdx.x; i < nW; i += blockDim.x ) 
    {
        src = pIn[i];
 	    switch (nType)
	    {
			case GPU_SCALAROPS_SUB:
				res.x = kernelScalarOpsSub(src.x, v0);
				res.y = kernelScalarOpsSub(src.y, v1);
				res.z = kernelScalarOpsSub(src.z, v2);
				break;

			case GPU_SCALAROPS_ADD:
				res.x = kernelScalarOpsAdd(src.x, v0);
				res.y = kernelScalarOpsAdd(src.y, v1);
				res.z = kernelScalarOpsAdd(src.z, v2);
				break;

			case GPU_SCALAROPS_NOT:
				res.x = ~src.x;
				res.y = ~src.y;
				res.z = ~src.z;
				break;

			case GPU_SCALAROPS_LUT:
				res.x = gpLUT[src.x];
				res.y = gpLUT[src.y];
				res.z = gpLUT[src.z];
				break;
		}
 		pOut[i] = res;
    }
}


__global__ void kernelScalarOpsRGBMask(uchar4 *pInput, uchar4 *pOutput, uchar4 *pMask, float v0, float v1, float v2, float v3, int nW, int nType)
{ 
    uchar4		src;
    uchar4		res;
 	uchar4		*pIn = pInput + blockIdx.x * nW;
	uchar4		*pOut = pOutput + blockIdx.x * nW;
	unsigned char *pM = (unsigned char *)pMask + blockIdx.x * nW;
  	unsigned char	mask;
  
    for ( int i = threadIdx.x; i < nW; i += blockDim.x ) 
    {
        src = pIn[i];
 		mask = pM[i];
		res = pOut[i];

		switch (nType)
	    {
			case GPU_SCALAROPS_SUB:
				kernelScalarOpsSubMask(src.x, v0, &(res.x), mask);
				kernelScalarOpsSubMask(src.y, v1, &(res.y), mask);
				kernelScalarOpsSubMask(src.z, v2, &(res.z), mask);
				break;

			case GPU_SCALAROPS_ADD:
				kernelScalarOpsAddMask(src.x, v0, &(res.x), mask);
				kernelScalarOpsAddMask(src.y, v1, &(res.y), mask);
				kernelScalarOpsAddMask(src.z, v2, &(res.z), mask);
				break;

			case GPU_SCALAROPS_NOT:
				if (mask != 0)
					res.x = ~src.x;
				else
					res.x = src.x;
				if (mask != 0)
					res.y = ~src.y;
				else
					res.y = src.y;
				if (mask != 0)
					res.z = ~src.z;
				else
					res.z = src.z;
				break;


			case GPU_SCALAROPS_LUT:
				res.x = gpLUT[src.x];
				res.y = gpLUT[src.y];
				res.z = gpLUT[src.z];
				break;
		}
 		pOut[i] = res;
    }
}

extern "C" int _RTGPUCreateLUT(unsigned char *LUT) 
{
	RTGPUTrace("RTGPUCreateLUT");
	RTGPUSafeCall(cudaMemcpyToSymbol(gpLUT, LUT, 256));
    return 1;
}


extern "C" int _RTGPUScalarOps(int srcSlot, RTGPUScalar scalar, int destSlot, int maskSlot, int type) 
{
	RTGPU_IMAGE *SI, *MI, *DI;

	RTGPUTrace("RTGPUScalarOps");

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	if (maskSlot != -1) {
		RTGPU_SLOTPTR(maskSlot, MI);
	} else {
		MI = NULL;
	}

	if (MI != NULL) {
		assert(SI->width == MI->width);
		assert(SI->height == MI->height);
	}

	_RTGPUSetupSlot(DI, SI->width, SI->height, SI->color);

	if (!SI->color) {
		if (MI != NULL)
			kernelScalarOpsGrayMask<<<SI->height, 32>>>(SI->image, DI->image, MI->image, scalar.val[0], SI->width, type);
		else
			kernelScalarOpsGray<<<SI->height, 32>>>(SI->image, DI->image, scalar.val[0], SI->width, type);
	} else {
		if (MI != NULL)
			kernelScalarOpsRGBMask<<<SI->height, 32>>>(SI->image, DI->image, MI->image, 
				scalar.val[0], scalar.val[1], scalar.val[2], scalar.val[3], SI->width, type);
		else
			kernelScalarOpsRGB<<<SI->height, 32>>>(SI->image, DI->image, 
				scalar.val[0], scalar.val[1], scalar.val[2], scalar.val[3], SI->width, type);
	}
    return 1;
}

