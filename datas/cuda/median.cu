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

#include "RTGPUDefs.h"

#define		MEDIAN_NTH		4

texture<unsigned char, 2> g_texGray;						// the gray char texture

__global__ void kernelMedianBlurGray(unsigned char *output, int w, int rad)
{
    unsigned char mArray[256];

    unsigned char *out = output + blockIdx.x * w;
	int		sum;
    int		kCount2 = ((rad *2 + 1) * (rad * 2 + 1) + 1) / 2;
	int		count;
	unsigned char val, minval, maxval;
   
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) 
    {
		for (int x = 0; x < 256; x++)
			mArray[x] = 0; 
		minval = 255;
		maxval = 0; 
		for (int y = -rad; y <= rad; y++)
		{
 			for (int x = -rad; x <= rad; x++)
 			{
 				val = tex2D( g_texGray, (float) (i+x), (float) (blockIdx.x+y));
  				mArray[val]++;
				if (val < minval)
  					minval = val;
  				if (val > maxval)
  					maxval = val; 
 			}
		}
		for (sum = 0, count = minval; count < maxval; count++)
		{
			sum += mArray[count];
			if (sum > kCount2)
				break; 
		}	
 		out[i] = (unsigned char)(count);
   }
}

__global__ void kernelMedianBlurGrayShared(unsigned char *output, int w, int rad)
{
    __shared__ unsigned short medianArray[256 * MEDIAN_NTH];

    unsigned char *out = output + blockIdx.x * w;
	int		sum;
    int		kCount2 = ((rad *2 + 1) * (rad * 2 + 1) + 1) / 2;
	int		count;
	unsigned char val, minval, maxval;
	
	unsigned short *mArray = medianArray + threadIdx.x;
   
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) 
    {
		#pragma unroll
		for (int x = 0; x < 256; x++)
			mArray[x * MEDIAN_NTH] = 0; 
		minval = 255;
		maxval = 0; 
		for (int y = -rad; y <= rad; y++)
		{
			#pragma unroll
 			for (int x = -rad; x <= rad; x++)
 			{
 				val = tex2D( g_texGray, (float) (i+x), (float) (blockIdx.x+y));
  				mArray[MEDIAN_NTH * val]++;
				if (val < minval)
  					minval = val;
  				if (val > maxval)
  					maxval = val; 
 			}
		}
		for (sum = 0, count = minval; count < maxval; count++)
		{
			sum += mArray[MEDIAN_NTH * count];
			if (sum > kCount2)
				break; 
		}
 		out[i] = (unsigned char)count;
   }
}


extern "C" int _RTGPUMedianBlur(int srcSlot, int destSlot, int rad) 
{
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	if (rad == 1)
		return _RTGPUMedianBlur3x3(srcSlot, destSlot);

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	RTGPUTrace("RTGPUMedianBlur");
	assert(!SI->color);

	if (rad == 0) {						// just copy image!
		RTGPUSafeCall(cudaMemcpy(DI->image, SI->image, SI->width * SI->height * sizeof(uchar4), cudaMemcpyDeviceToDevice));
		return 1;
	}

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_texGray, SI->image, desc, SI->width, SI->height, SI->width));

	kernelMedianBlurGrayShared<<<SI->height, MEDIAN_NTH>>>((unsigned char *)DI->image, SI->width, rad);
	RTGPUSafeCall(cudaUnbindTexture(g_texGray));
		
    return 1;
}


