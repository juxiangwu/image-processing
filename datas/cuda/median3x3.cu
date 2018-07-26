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

#define		NTH			32

texture<unsigned char, 2> g_texGray;						// the gray char texture

__global__ void kernelMedianBlur3x3GrayShared(unsigned char *output, int w)
{
	__shared__ unsigned char medianArray[256 * NTH];
	unsigned char *mArray = medianArray + threadIdx.x;

    unsigned char *out = output + blockIdx.x * w;
	int sum;
	int count;
	int pix = (w + NTH - 1)/NTH;
	int iStart = pix * threadIdx.x;
	int iStop = iStart + pix;
	unsigned char minval, maxval;

	unsigned char	px00, px01, px02, px10, px11, px12, px20, px21, px22;

	if (iStop > w)
		iStop = w;

	#pragma unroll
	for (int x = 0; x < 256; x++)
		mArray[NTH * x] = 0;

	px00 = tex2D(g_texGray, (float) (iStart-1), (float) (blockIdx.x-1));	// preset pixel values
	px01 = tex2D(g_texGray, (float) (iStart-1), (float) (blockIdx.x-0));
	px02 = tex2D(g_texGray, (float) (iStart-1), (float) (blockIdx.x+1));
	px10 = tex2D(g_texGray, (float) (iStart), (float) (blockIdx.x-1));	
	px11 = tex2D(g_texGray, (float) (iStart), (float) (blockIdx.x-0));
	px12 = tex2D(g_texGray, (float) (iStart), (float) (blockIdx.x+1));
	px20 = tex2D(g_texGray, (float) (iStart+1), (float) (blockIdx.x-1));	
	px21 = tex2D(g_texGray, (float) (iStart+1), (float) (blockIdx.x-0));
	px22 = tex2D(g_texGray, (float) (iStart+1), (float) (blockIdx.x+1));

	mArray[NTH * px00]++;						// compute first median
	mArray[NTH * px01]++;
	mArray[NTH * px02]++;
	mArray[NTH * px10]++;
	mArray[NTH * px11]++;
	mArray[NTH * px12]++;
	mArray[NTH * px20]++;
	mArray[NTH * px21]++;
	mArray[NTH * px22]++;

	minval = 255;
	maxval = 0;
	if (px00 < minval)
		minval = px00;
	if (px00 > maxval)
		maxval = px00;

	if (px01 < minval)
		minval = px01;
	if (px01 > maxval)
		maxval = px01;

	if (px02 < minval)
		minval = px02;
	if (px02 > maxval)
		maxval = px02;

	if (px10 < minval)
		minval = px10;
	if (px10 > maxval)
		maxval = px10;

	if (px11 < minval)
		minval = px11;
	if (px11 > maxval)
		maxval = px11;

	if (px12 < minval)
		minval = px12;
	if (px12 > maxval)
		maxval = px12;

	if (px20 < minval)
		minval = px20;
	if (px20 > maxval)
		maxval = px20;

	if (px21 < minval)
		minval = px21;
	if (px21 > maxval)
		maxval = px21;

	if (px22 < minval)
		minval = px22;
	if (px22 > maxval)
		maxval = px22;

	if (minval == maxval){
		out[iStart] = minval;
	} else {
		for (sum = 0, count = minval; count < maxval; count++){
			sum += mArray[NTH * count];
			if (sum > 4)
				break; 
		}
 		out[iStart] = (unsigned char)count;
	}

	for ( int i = iStart+1; i < iStop; i++ )  {

//		replace correct set of column pixels

		switch (i % 3) {
			case 0:
 				mArray[NTH * px00]--;			// take out old values
				mArray[NTH * px01]--;
				mArray[NTH * px02]--;
				px00 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x-1)); // load new values
				px01 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x-0));
				px02 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x+1));
 				mArray[NTH * px00]++;			// add in new values
				mArray[NTH * px01]++;
				mArray[NTH * px02]++;
				break;

			case 1:
 				mArray[NTH * px10]--;
				mArray[NTH * px11]--;
				mArray[NTH * px12]--;
				px10 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x-1));
				px11 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x-0));
				px12 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x+1));
 				mArray[NTH * px10]++;
				mArray[NTH * px11]++;
				mArray[NTH * px12]++;
				break;

			case 2:
 				mArray[NTH * px20]--;
				mArray[NTH * px21]--;
				mArray[NTH * px22]--;
				px20 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x-1));
				px21 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x-0));
				px22 = tex2D( g_texGray, (float) (i + 1), (float) (blockIdx.x+1));
 				mArray[NTH * px20]++;
				mArray[NTH * px21]++;
				mArray[NTH * px22]++;
				break;
		}
		minval = 255;
		maxval = 0;
		if (px00 < minval)
			minval = px00;
		if (px00 > maxval)
			maxval = px00;

		if (px01 < minval)
			minval = px01;
		if (px01 > maxval)
			maxval = px01;

		if (px02 < minval)
			minval = px02;
		if (px02 > maxval)
			maxval = px02;

		if (px10 < minval)
			minval = px10;
		if (px10 > maxval)
			maxval = px10;

		if (px11 < minval)
			minval = px11;
		if (px11 > maxval)
			maxval = px11;

		if (px12 < minval)
			minval = px12;
		if (px12 > maxval)
			maxval = px12;

		if (px20 < minval)
			minval = px20;
		if (px20 > maxval)
			maxval = px20;

		if (px21 < minval)
			minval = px21;
		if (px21 > maxval)
			maxval = px21;

		if (px22 < minval)
			minval = px22;
		if (px22 > maxval)
			maxval = px22;
		if (minval == maxval) {
 			out[i] = minval;
		} else {
			for (sum = 0, count = minval; count < maxval; count++) {
				sum += mArray[NTH * count];
				if (sum > 4)
					break; 
			}
	 		out[i] = (unsigned char)count;
		}
   }
}


extern "C" int _RTGPUMedianBlur3x3(int srcSlot, int destSlot) 
{
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	RTGPUTrace("RTGPUMedianBlur3x3");
	
	assert(!SI->color);

	_RTGPUSetupSlot(DI, SI->width, SI->height, false);

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_texGray, SI->image, desc, SI->width, SI->height, SI->width));
	
	kernelMedianBlur3x3GrayShared<<<SI->height, NTH>>>((unsigned char *)DI->image, SI->width);
	
	RTGPUSafeCall(cudaUnbindTexture(g_texGray));
		
    return 1;
}


