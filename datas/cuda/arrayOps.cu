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

//	kernelArrayOpsAbsDiff is a saturating scalar difference function. *nRes = abs(nA - nB).
//

__device__ void kernelArrayOpsAbsDiff(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = abs(nA - nB);
}

//	kernelArrayOpsCmpEQ sets nRes to 0xff if nA == nB
//

__device__ void kernelArrayOpsCmpEQ(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = nA == nB ? 0xff : 0;
}

//	kernelArrayOpsCmpGT sets nRes to 0xff if nA > nB
//

__device__ void kernelArrayOpsCmpGT(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = nA > nB ? 0xff : 0;
}

//	kernelArrayOpsCmpGE sets nRes to 0xff if nA >= nB
//

__device__ void kernelArrayOpsCmpGE(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = nA >= nB ? 0xff : 0;
}

//	kernelArrayOpsCmpLT sets nRes to 0xff if nA < nB
//

__device__ void kernelArrayOpsCmpLT(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = nA < nB ? 0xff : 0;
}

//	kernelArrayOpsCmpLE sets nRes to 0xff if nA <= nB
//

__device__ void kernelArrayOpsCmpLE(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = nA <= nB ? 0xff : 0;
}

//	kernelArrayOpsCmpNE sets nRes to 0xff if nA != nB
//

__device__ void kernelArrayOpsCmpNE(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = nA != nB ? 0xff : 0;
}

//	kernelArrayOpsOr sets nRes = nA | nB
//

__device__ void kernelArrayOpsOr(unsigned char nA, unsigned char nB, unsigned char *pnRes, int nMask)
{
	if (nMask == 0)
		return;
	*pnRes = nA | nB;
}

__global__ void kernelArrayOpsGray(uchar4 *pIA, uchar4 *pIB, uchar4 *pOutput, uchar4 *pMask, int nW, int nType)
{ 
    uchar4		srcA, srcB;
    uchar4		res;
 	uchar4		*pM = pMask + blockIdx.x * nW / 4;
 	uchar4		*pA = pIA + blockIdx.x * nW / 4;
 	uchar4		*pB = pIB + blockIdx.x * nW / 4;
 	uchar4		*pOut = pOutput + blockIdx.x * nW / 4;
  	uchar4		mask;
	int			i, j;
 
    for (i = threadIdx.x, j = 4 * threadIdx.x; j < nW; i += blockDim.x, j += 4 * blockDim.x ) 
    {
        srcA = pA[i];
        srcB = pB[i];
 		if (pMask != NULL)
		{
  			mask = pM[i];
			res = pOut[i];
		}
		else
		{
			mask.x = mask.y = mask.z = mask.w = 1;
		}
	    
	    switch (nType)
	    {
			case GPU_ARRAYOPS_ABSDIFF:
				kernelArrayOpsAbsDiff(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsAbsDiff(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsAbsDiff(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsAbsDiff(srcA.w, srcB.w, &(res.w), mask.w);
				break;

			case GPU_ARRAYOPS_CMPEQ:
				kernelArrayOpsCmpEQ(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsCmpEQ(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsCmpEQ(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsCmpEQ(srcA.w, srcB.w, &(res.w), mask.w);
				break;

			case GPU_ARRAYOPS_CMPGT:
				kernelArrayOpsCmpGT(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsCmpGT(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsCmpGT(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsCmpGT(srcA.w, srcB.w, &(res.w), mask.w);
				break;

			case GPU_ARRAYOPS_CMPGE:
				kernelArrayOpsCmpGE(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsCmpGE(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsCmpGE(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsCmpGE(srcA.w, srcB.w, &(res.w), mask.w);
				break;

			case GPU_ARRAYOPS_CMPLE:
				kernelArrayOpsCmpLE(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsCmpLE(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsCmpLE(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsCmpLE(srcA.w, srcB.w, &(res.w), mask.w);
				break;

			case GPU_ARRAYOPS_CMPLT:
				kernelArrayOpsCmpLT(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsCmpLT(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsCmpLT(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsCmpLT(srcA.w, srcB.w, &(res.w), mask.w);
				break;

			case GPU_ARRAYOPS_CMPNE:
				kernelArrayOpsCmpNE(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsCmpNE(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsCmpNE(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsCmpNE(srcA.w, srcB.w, &(res.w), mask.w);
				break;

			case GPU_ARRAYOPS_OR:
				kernelArrayOpsOr(srcA.x, srcB.x, &(res.x), mask.x);
				kernelArrayOpsOr(srcA.y, srcB.y, &(res.y), mask.y);
				kernelArrayOpsOr(srcA.z, srcB.z, &(res.z), mask.z);
				kernelArrayOpsOr(srcA.w, srcB.w, &(res.w), mask.w);
				break;

		}
 		pOut[i] = res;
    }
}

__global__ void kernelArrayOpsRGB(uchar4 *pIA, uchar4 *pIB, uchar4 *pOutput, uchar4 *pMask, int nW, int nType)
{ 
    uchar4		srcA, srcB;
    uchar4		res;
 	uchar4		*pA = pIA + blockIdx.x * nW;
 	uchar4		*pB = pIB + blockIdx.x * nW;
 	uchar4		*pOut = pOutput + blockIdx.x * nW;
 	unsigned char *pM = (unsigned char *)pMask + blockIdx.x * nW;
  	unsigned char	mask;
   
    for ( int i = threadIdx.x; i < nW; i += blockDim.x ) 
    {
        srcA = pA[i];
        srcB = pB[i];
 		if (pMask != NULL)
		{
  			mask = pM[i];
			res = pOut[i];
		}
		else
		{
			mask = 1;
		}
 	    
	    switch (nType)
	    {
			case GPU_ARRAYOPS_ABSDIFF:
				kernelArrayOpsAbsDiff(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsAbsDiff(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsAbsDiff(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsAbsDiff(srcA.w, srcB.w, &(res.w), mask);
				break;

			case GPU_ARRAYOPS_CMPEQ:
				kernelArrayOpsCmpEQ(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsCmpEQ(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsCmpEQ(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsCmpEQ(srcA.w, srcB.w, &(res.w), mask);
				break;

			case GPU_ARRAYOPS_CMPGT:
				kernelArrayOpsCmpGT(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsCmpGT(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsCmpGT(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsCmpGT(srcA.w, srcB.w, &(res.w), mask);
				break;

			case GPU_ARRAYOPS_CMPGE:
				kernelArrayOpsCmpGE(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsCmpGE(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsCmpGE(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsCmpGE(srcA.w, srcB.w, &(res.w), mask);
				break;

			case GPU_ARRAYOPS_CMPLE:
				kernelArrayOpsCmpLE(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsCmpLE(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsCmpLE(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsCmpLE(srcA.w, srcB.w, &(res.w), mask);
				break;

			case GPU_ARRAYOPS_CMPLT:
				kernelArrayOpsCmpLT(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsCmpLT(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsCmpLT(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsCmpLT(srcA.w, srcB.w, &(res.w), mask);
				break;

			case GPU_ARRAYOPS_CMPNE:
				kernelArrayOpsCmpNE(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsCmpNE(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsCmpNE(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsCmpNE(srcA.w, srcB.w, &(res.w), mask);
				break;

			case GPU_ARRAYOPS_OR:
				kernelArrayOpsOr(srcA.x, srcB.x, &(res.x), mask);
				kernelArrayOpsOr(srcA.y, srcB.y, &(res.y), mask);
				kernelArrayOpsOr(srcA.z, srcB.z, &(res.z), mask);
				kernelArrayOpsOr(srcA.w, srcB.w, &(res.w), mask);
				break;
		}
 		pOut[i] = res;
    }
}


extern "C" int _RTGPUArrayOps(int srcSlotA, int srcSlotB, int destSlot, int maskSlot, int type) 
{
	RTGPU_IMAGE *SIA, *SIB, *MI, *DI;
	
	RTGPUTrace("RTGPUArrayOps");

	RTGPU_SLOTPTR(srcSlotA, SIA);
	RTGPU_SLOTPTR(srcSlotB, SIB);
	RTGPU_SLOTPTR(destSlot, DI);

	if (maskSlot != -1) {
		RTGPU_SLOTPTR(maskSlot, MI);
	} else {
		MI = NULL;
	}

	assert(SIA->width == SIB->width);
	assert(SIA->height == SIB->height);
	assert(SIA->color == SIB->color);
	if (MI != NULL) {
		assert(SIA->width == MI->width);
		assert(SIA->height == MI->height);
	}

	if (!SIA->color) {
		if (MI != NULL)
			kernelArrayOpsGray<<<SIA->height, 32>>>(SIA->image, SIB->image, DI->image, MI->image, SIA->width, type);
		else
			kernelArrayOpsGray<<<SIA->height, 32>>>(SIA->image, SIB->image, DI->image, NULL, SIA->width, type);
	} else {
		if (MI != NULL)
			kernelArrayOpsRGB<<<SIA->height, 32>>>(SIA->image, SIB->image, DI->image, MI->image, SIA->width, type);
		else
			kernelArrayOpsRGB<<<SIA->height, 32>>>(SIA->image, SIB->image, DI->image, NULL, SIA->width, type);
	}
    return 1;
}