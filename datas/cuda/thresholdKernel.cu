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

texture<unsigned char, 2> g_tex1Gray;					// the gray src image
texture<unsigned char, 2> g_tex2Gray;					// the gray inter

__constant__ unsigned char g_threshTab[768];			// threshold table


__global__ void kernelThreshold(unsigned char *output, int w)
{ 
    unsigned char *pRes = output + blockIdx.x * w;
    unsigned char src;
    uchar4 res;
    
    #pragma unroll
    for ( int i = 4 * threadIdx.x; i < w; i += 4 * blockDim.x ) 
    {
        src = tex2D(g_tex1Gray, (float) i, (float) blockIdx.x);
 		res.x = g_threshTab[src];
        src = tex2D(g_tex1Gray, (float) i+1, (float) blockIdx.x);
 		res.y = g_threshTab[src];
        src = tex2D(g_tex1Gray, (float) i+2, (float) blockIdx.x);
 		res.z = g_threshTab[src];
        src = tex2D(g_tex1Gray, (float) i+3, (float) blockIdx.x);
 		res.w = g_threshTab[src];
 		*(uchar4 *)(pRes+i) = res;
    }
}


__global__ void kernelAdaptiveThreshold(unsigned char *output, int w)
{ 
    unsigned char	*pRes = output + blockIdx.x * w;
    unsigned char	src, inter;
    
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) 
    {
        src = tex2D(g_tex1Gray, (float) i, (float) blockIdx.x);
        inter = tex2D(g_tex2Gray, (float) i, (float) blockIdx.x);
		
		pRes[i] = g_threshTab[src - inter + 255];
    }
}


extern "C" int _RTGPUThreshold(int srcSlot, int destSlot, int thresh, int maxVal, int type) 
{
	unsigned char tab[256];    
	int i;

	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	RTGPUTrace("RTGPUThreshold");

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	assert(!SI->color);

	_RTGPUSetupSlot(DI, SI->width, SI->height, false);

	switch (type)
	{
		case GPU_THRESH_BINARY:
			for( i = 0; i <= thresh; i++ )
				tab[i] = 0;
			for( ; i < 256; i++ )
				tab[i] = maxVal;
			break;
			
		case GPU_THRESH_BINARY_INV:
			for( i = 0; i <= thresh; i++ )
				tab[i] = maxVal;
			for( ; i < 256; i++ )
				tab[i] = 0;
			break;
			
		case GPU_THRESH_TRUNC:
			for( i = 0; i <= thresh; i++ )
				tab[i] = (unsigned char)i;
			for( ; i < 256; i++ )
				tab[i] = thresh;
			break;
			
		case GPU_THRESH_TOZERO:
			for( i = 0; i <= thresh; i++ )
				tab[i] = 0;
			for( ; i < 256; i++ )
				tab[i] = (unsigned char)i;
			break;
			
		case GPU_THRESH_TOZERO_INV:
			for( i = 0; i <= thresh; i++ )
				tab[i] = (unsigned char)i;
			for( ; i < 256; i++ )
				tab[i] = 0;
			break;
 
 		default:
			RTGPUError("Unknown/unsupported threshold type" );
			return 0;
	}
	
	cudaMemcpyToSymbol(g_threshTab, tab, 256);

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex1Gray, SI->image, desc, SI->width, SI->height, SI->width));

	kernelThreshold<<<SI->height, 32>>>((unsigned char *)DI->image, SI->width);

	RTGPUSafeCall(cudaUnbindTexture(g_tex1Gray));

    return 1;
}

extern "C" int _RTGPUAdaptiveThreshold(int srcSlot, int destSlot, int maxVal, int method, int type, int rad, int delta) 
{
	RTGPU_IMAGE	*II;
	
	unsigned char tab[768];    
	int i;
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	RTGPUTrace("RTGPUAdaptiveThreshold");

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	assert(!SI->color);

	_RTGPUSetupSlot(DI, SI->width, SI->height, false);

	II = g_images + INTERNAL_GPU_SLOT0;

	switch (type)
	{
		case GPU_THRESH_BINARY:
			for( i = 0; i < 768; i++ )
				tab[i] = (unsigned char)(i - 255 > -delta ? maxVal : 0);
			break;
 
		case GPU_THRESH_BINARY_INV:
			for( i = 0; i < 768; i++ )
				tab[i] = (unsigned char)(i - 255 <= -delta ? maxVal : 0);
			break;
							
		default:
			RTGPUError("Unknown/unsupported threshold type" );
			return 0;
	}
	
	if (!_RTGPUBoxFilter(srcSlot, INTERNAL_GPU_SLOT0, rad, true))
		return 0;

	cudaMemcpyToSymbol(g_threshTab, tab, 768);

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex1Gray, SI->image, desc, SI->width, SI->height, SI->width));
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex2Gray, II->image, desc, SI->width, SI->height, SI->width));
	
	kernelAdaptiveThreshold<<<SI->height, 256>>>((unsigned char *)DI->image, SI->width);

	RTGPUSafeCall(cudaUnbindTexture(g_tex1Gray));
	RTGPUSafeCall(cudaUnbindTexture(g_tex2Gray));

    return 1;
}

