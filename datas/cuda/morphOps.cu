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

#include "RTGPUDefs.h"

#define		NTH			32

texture<unsigned char, 2> gpTexGray;						// the gray char texture
texture<uchar4, 2> gpTexRGB;								// the RGB texture


__global__ void kernelMorphOps3x3Gray(unsigned char *pOutput, int nW, int nMode)
{
    unsigned char *pOut = pOutput + blockIdx.x * nW;
	int		nPix = (nW + NTH - 1)/NTH;
	int		iStart = nPix * threadIdx.x;
	int		iStop = iStart + nPix;
	unsigned char minval, maxval;

	unsigned char	px00, px01, px02, px10, px11, px12, px20, px21, px22;

	if (iStop > nW)
		iStop = nW;

	px00 = tex2D( gpTexGray, (float) (iStart-1), (float) (blockIdx.x-1));	// preset pixel values
	px01 = tex2D( gpTexGray, (float) (iStart-1), (float) (blockIdx.x-0));
	px02 = tex2D( gpTexGray, (float) (iStart-1), (float) (blockIdx.x+1));
	px10 = tex2D( gpTexGray, (float) (iStart), (float) (blockIdx.x-1));	
	px11 = tex2D( gpTexGray, (float) (iStart), (float) (blockIdx.x-0));
	px12 = tex2D( gpTexGray, (float) (iStart), (float) (blockIdx.x+1));
	px20 = tex2D( gpTexGray, (float) (iStart+1), (float) (blockIdx.x-1));	
	px21 = tex2D( gpTexGray, (float) (iStart+1), (float) (blockIdx.x-0));
	px22 = tex2D( gpTexGray, (float) (iStart+1), (float) (blockIdx.x+1));

	minval = 255;
	maxval = 0;

	switch (nMode)
	{
		case GPU_MORPHOPS_DILATE:
			RTGPU_MAX(px00, maxval);
			RTGPU_MAX(px01, maxval);
			RTGPU_MAX(px02, maxval);
			RTGPU_MAX(px10, maxval);
			RTGPU_MAX(px11, maxval);
			RTGPU_MAX(px12, maxval);
			RTGPU_MAX(px20, maxval);
			RTGPU_MAX(px21, maxval);
			RTGPU_MAX(px22, maxval);
			pOut[iStart] = maxval;
			break;

		case GPU_MORPHOPS_ERODE:
			RTGPU_MIN(px00, minval);
			RTGPU_MIN(px01, minval);
			RTGPU_MIN(px02, minval);
			RTGPU_MIN(px10, minval);
			RTGPU_MIN(px11, minval);
			RTGPU_MIN(px12, minval);
			RTGPU_MIN(px20, minval);
			RTGPU_MIN(px21, minval);
			RTGPU_MIN(px22, minval);
			pOut[iStart] = minval;
			break;
	}

	for ( int i = iStart+1; i < iStop; i++ ) 
    {

//		replace correct set of column pixels

		switch (i % 3)
		{
			case 0:
				px00 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x-1)); // load new values
				px01 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x-0));
				px02 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x+1));
				break;

			case 1:
				px10 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x-1));
				px11 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x-0));
				px12 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x+1));
				break;

			case 2:
				px20 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x-1));
				px21 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x-0));
				px22 = tex2D( gpTexGray, (float) (i + 1), (float) (blockIdx.x+1));
				break;
		}
		minval = 255;
		maxval = 0;

		switch (nMode)
		{
			case GPU_MORPHOPS_DILATE:
				RTGPU_MAX(px00, maxval);
				RTGPU_MAX(px01, maxval);
				RTGPU_MAX(px02, maxval);
				RTGPU_MAX(px10, maxval);
				RTGPU_MAX(px11, maxval);
				RTGPU_MAX(px12, maxval);
				RTGPU_MAX(px20, maxval);
				RTGPU_MAX(px21, maxval);
				RTGPU_MAX(px22, maxval);
				pOut[i] = maxval;
				break;

			case GPU_MORPHOPS_ERODE:
				RTGPU_MIN(px00, minval);
				RTGPU_MIN(px01, minval);
				RTGPU_MIN(px02, minval);
				RTGPU_MIN(px10, minval);
				RTGPU_MIN(px11, minval);
				RTGPU_MIN(px12, minval);
				RTGPU_MIN(px20, minval);
				RTGPU_MIN(px21, minval);
				RTGPU_MIN(px22, minval);
				pOut[i] = minval;
				break;
		}
	}
}

__global__ void kernelMorphOps3x3RGB(uchar4 *pOutput, int nW, int nMode)
{
    uchar4	*pOut = pOutput + blockIdx.x * nW;
	int		nPix = (nW + NTH - 1)/NTH;
	int		iStart = nPix * threadIdx.x;
	int		iStop = iStart + nPix;
	uchar4	minval, maxval;

	uchar4	px00, px01, px02, px10, px11, px12, px20, px21, px22;

	if (iStop > nW)
		iStop = nW;

	px00 = tex2D( gpTexRGB, (float) (iStart-1), (float) (blockIdx.x-1));	// preset pixel values
	px01 = tex2D( gpTexRGB, (float) (iStart-1), (float) (blockIdx.x-0));
	px02 = tex2D( gpTexRGB, (float) (iStart-1), (float) (blockIdx.x+1));
	px10 = tex2D( gpTexRGB, (float) (iStart), (float) (blockIdx.x-1));	
	px11 = tex2D( gpTexRGB, (float) (iStart), (float) (blockIdx.x-0));
	px12 = tex2D( gpTexRGB, (float) (iStart), (float) (blockIdx.x+1));
	px20 = tex2D( gpTexRGB, (float) (iStart+1), (float) (blockIdx.x-1));	
	px21 = tex2D( gpTexRGB, (float) (iStart+1), (float) (blockIdx.x-0));
	px22 = tex2D( gpTexRGB, (float) (iStart+1), (float) (blockIdx.x+1));

	minval.x = minval.y = minval.z = minval.w = 255;
	maxval.x = maxval.y = maxval.z = maxval.w = 0;

	switch (nMode)
	{
		case GPU_MORPHOPS_DILATE:
			RTGPU_MAX4(px00, maxval);
			RTGPU_MAX4(px01, maxval);
			RTGPU_MAX4(px02, maxval);
			RTGPU_MAX4(px10, maxval);
			RTGPU_MAX4(px11, maxval);
			RTGPU_MAX4(px12, maxval);
			RTGPU_MAX4(px20, maxval);
			RTGPU_MAX4(px21, maxval);
			RTGPU_MAX4(px22, maxval);
			pOut[iStart] = maxval;
			break;

		case GPU_MORPHOPS_ERODE:
			RTGPU_MIN4(px00, minval);
			RTGPU_MIN4(px01, minval);
			RTGPU_MIN4(px02, minval);
			RTGPU_MIN4(px10, minval);
			RTGPU_MIN4(px11, minval);
			RTGPU_MIN4(px12, minval);
			RTGPU_MIN4(px20, minval);
			RTGPU_MIN4(px21, minval);
			RTGPU_MIN4(px22, minval);
			pOut[iStart] = minval;
			break;
	}

	for ( int i = iStart+1; i < iStop; i++ ) 
    {

//		replace correct set of column pixels

		switch (i % 3)
		{
			case 0:
				px00 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x-1)); // load new values
				px01 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x-0));
				px02 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x+1));
				break;

			case 1:
				px10 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x-1));
				px11 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x-0));
				px12 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x+1));
				break;

			case 2:
				px20 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x-1));
				px21 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x-0));
				px22 = tex2D( gpTexRGB, (float) (i + 1), (float) (blockIdx.x+1));
				break;
		}
		minval.x = minval.y = minval.z = minval.w = 255;
		maxval.x = maxval.y = maxval.z = maxval.w = 0;

		switch (nMode)
		{
			case GPU_MORPHOPS_DILATE:
				RTGPU_MAX4(px00, maxval);
				RTGPU_MAX4(px01, maxval);
				RTGPU_MAX4(px02, maxval);
				RTGPU_MAX4(px10, maxval);
				RTGPU_MAX4(px11, maxval);
				RTGPU_MAX4(px12, maxval);
				RTGPU_MAX4(px20, maxval);
				RTGPU_MAX4(px21, maxval);
				RTGPU_MAX4(px22, maxval);
				pOut[i] = maxval;
				break;

			case GPU_MORPHOPS_ERODE:
				RTGPU_MIN4(px00, minval);
				RTGPU_MIN4(px01, minval);
				RTGPU_MIN4(px02, minval);
				RTGPU_MIN4(px10, minval);
				RTGPU_MIN4(px11, minval);
				RTGPU_MIN4(px12, minval);
				RTGPU_MIN4(px20, minval);
				RTGPU_MIN4(px21, minval);
				RTGPU_MIN4(px22, minval);
				pOut[i] = minval;
				break;
		}
	}
}

extern "C" int _RTGPUMorphOps3x3(int srcSlot, int destSlot, int mode) 
{
	RTGPU_IMAGE	*SI, *DI;
	RTGPUTrace("RTGPUMorphOps3x3");
    cudaChannelFormatDesc desc;

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	_RTGPUSetupSlot(DI, SI->width, SI->height, SI->color);

	if (!SI->color) {
		desc = cudaCreateChannelDesc<unsigned char>();
		RTGPUSafeCall(cudaBindTexture2D(NULL, gpTexGray, SI->image, desc, SI->width, SI->height, SI->width));
		kernelMorphOps3x3Gray<<<SI->height, NTH>>>((unsigned char *)DI->image, SI->width, mode);
		RTGPUSafeCall(cudaUnbindTexture(gpTexGray));
	} else {
		desc = cudaCreateChannelDesc<uchar4>();
		RTGPUSafeCall(cudaBindTexture2D(NULL, gpTexRGB, SI->image, desc, SI->width, SI->height, SI->width * 4));
		kernelMorphOps3x3RGB<<<SI->height, NTH>>>((uchar4 *)DI->image, SI->width, mode);
		RTGPUSafeCall(cudaUnbindTexture(gpTexRGB));
	}
		
    return 1;
}


