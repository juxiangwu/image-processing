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

texture<unsigned char, 2> g_tex0;						// the char texture

//-----------------------------------------------------------------------------
//
//	2D Sobel operator

__device__ unsigned char kernelComputeSobel(
			 unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr // lower right
            )
{
	int	horiz, vert, sum;
	
    horiz = ur + 2*mr + lr - ul - 2*ml - ll;
    vert = ul + 2*um + ur - ll - 2*lm - lr;
    sum = (short) ((abs(horiz)+abs(vert)));
    if ( sum < 0 ) 
		sum = 0; 
	else 
		if ( sum > 0xff ) 
			sum = 0xff;
	return (unsigned char)sum;
}

__global__ void kernelSobelFilter1( unsigned char *output, int w)
{ 
    unsigned char *sobel = output + blockIdx.x * w;
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) 
    {
        unsigned char pix00 = tex2D( g_tex0, (float) i-1, (float) blockIdx.x-1 );
		unsigned char pix01 = tex2D( g_tex0, (float) i+0, (float) blockIdx.x-1 );
        unsigned char pix02 = tex2D( g_tex0, (float) i+1, (float) blockIdx.x-1 );
        unsigned char pix10 = tex2D( g_tex0, (float) i-1, (float) blockIdx.x+0 );
        unsigned char pix12 = tex2D( g_tex0, (float) i+1, (float) blockIdx.x+0 );
        unsigned char pix20 = tex2D( g_tex0, (float) i-1, (float) blockIdx.x+1 );
        unsigned char pix21 = tex2D( g_tex0, (float) i+0, (float) blockIdx.x+1 );
        unsigned char pix22 = tex2D( g_tex0, (float) i+1, (float) blockIdx.x+1 );
        sobel[i] = kernelComputeSobel(pix00, pix01, pix02, 
                                 pix10,		   pix12,
                                 pix20, pix21, pix22);
    }
}


__global__ void kernelSobelFilter4(unsigned char *output, int w)
{ 
    unsigned char *sobel = output + blockIdx.x * w;
    uchar4	res;
    
    for ( int i = 4 * threadIdx.x; i < w; i += 4 * blockDim.x ) 
    {
        unsigned char pix00 = tex2D( g_tex0, (float) i-1, (float) blockIdx.x-1 );
		unsigned char pix01 = tex2D( g_tex0, (float) i+0, (float) blockIdx.x-1 );
        unsigned char pix02 = tex2D( g_tex0, (float) i+1, (float) blockIdx.x-1 );
        unsigned char pix10 = tex2D( g_tex0, (float) i-1, (float) blockIdx.x+0 );
        unsigned char pix11 = tex2D( g_tex0, (float) i+0, (float) blockIdx.x+0 );
        unsigned char pix12 = tex2D( g_tex0, (float) i+1, (float) blockIdx.x+0 );
        unsigned char pix20 = tex2D( g_tex0, (float) i-1, (float) blockIdx.x+1 );
        unsigned char pix21 = tex2D( g_tex0, (float) i+0, (float) blockIdx.x+1 );
        unsigned char pix22 = tex2D( g_tex0, (float) i+1, (float) blockIdx.x+1 );
        res.x = kernelComputeSobel(pix00, pix01, pix02, 
                                 pix10,		   pix12,
                                 pix20, pix21, pix22);
                                 
		pix00 = pix01;
		pix10 = pix11;
		pix20 = pix21;
		pix01 = pix02;
		pix11 = pix12;
		pix21 = pix22;
		pix02 = tex2D( g_tex0, (float) i+1+1, (float) blockIdx.x-1 );
		pix12 = tex2D( g_tex0, (float) i+1+1, (float) blockIdx.x+0 );
		pix22 = tex2D( g_tex0, (float) i+1+1, (float) blockIdx.x+1 );
        res.y = kernelComputeSobel(pix00, pix01, pix02, 
                                 pix10,		   pix12,
                                 pix20, pix21, pix22);
                                 
       	pix00 = pix01;
		pix10 = pix11;
		pix20 = pix21;
		pix01 = pix02;
		pix11 = pix12;
		pix21 = pix22;
		pix02 = tex2D( g_tex0, (float) i+1+2, (float) blockIdx.x-1 );
		pix12 = tex2D( g_tex0, (float) i+1+2, (float) blockIdx.x+0 );
		pix22 = tex2D( g_tex0, (float) i+1+2, (float) blockIdx.x+1 );
        res.z = kernelComputeSobel(pix00, pix01, pix02, 
                                 pix10,		   pix12,
                                 pix20, pix21, pix22);
                                 
        pix00 = pix01;
		pix10 = pix11;
		pix20 = pix21;
		pix01 = pix02;
		pix11 = pix12;
		pix21 = pix22;
		pix02 = tex2D( g_tex0, (float) i+1+3, (float) blockIdx.x-1 );
		pix12 = tex2D( g_tex0, (float) i+1+3, (float) blockIdx.x+0 );
		pix22 = tex2D( g_tex0, (float) i+1+3, (float) blockIdx.x+1 );
        res.w = kernelComputeSobel(pix00, pix01, pix02, 
                                 pix10,		   pix12,
                                 pix20, pix21, pix22);
       *(uchar4 *)(sobel + i) = res;
    }
}



extern "C" int _RTGPUSobel(int srcSlot, int destSlot) 
{
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);
	
	RTGPUTrace("RTGPUSobel");

	assert(!SI->color);

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex0, SI->image, desc, DI->width, DI->height, DI->width));

	kernelSobelFilter4<<<DI->height, 256>>>((unsigned char *)DI->image, DI->width);

	RTGPUSafeCall(cudaUnbindTexture(g_tex0));
		
    return 1;
}
