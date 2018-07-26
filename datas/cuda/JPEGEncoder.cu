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
//	Derived from the following:

/****************************************************************************
*	Author:			Dr. Tony Lin											*
*	Email:			lintong@cis.pku.edu.cn									*
*	Release Date:	Dec. 2002												*
*																			*
*	Name:			TonyJpegLib, rewritten from IJG codes					*
*	Source:			IJG v.6a JPEG LIB										*
*	Purpose��		Support real jpeg file, with readable code				*
*																			*
*	Acknowlegement:	Thanks for great IJG, and Chris Losinger				*
*																			*
*	Legal Issues:	(almost same as IJG with followings)					*
*																			*
*	1. We don't promise that this software works.							*
*	2. You can use this software for whatever you want.						*
*	You don't have to pay.													*
*	3. You may not pretend that you wrote this software. If you use it		*
*	in a program, you must acknowledge somewhere. That is, please			*
*	metion IJG, and Me, Dr. Tony Lin.										*
*																			*
*****************************************************************************/

#include "RTGPUDefs.h"

#define	BLOCK16	16					// x and y size of the 16 x 16 block obviously

#define	DCTBLOCKS	6				// number of blocks to DCT in a YCBCR block
#define	DCTSIZE		8				// dimension of the DCT block
#define	YCBCRBLOCK	(256 + 64 + 64)	// total length of the YCBCR block
#define	CBOFFSET	256				// offset to Cb part
#define	CROFFSET	320				// offset to Cr part


texture<uchar4, 2>			gpTexRGB;				// the rgb texture
texture<int>				gpTexYCBCR;				// the YCBCR texture

__constant__	int gcRToY[256];
__constant__	int gcGToY[256];
__constant__	int gcBToY[256];
__constant__	int gcRToCb[256];
__constant__	int gcGToCb[256];
__constant__	int gcBToCb[256];
__constant__	int gcRToCr[256];
__constant__	int gcGToCr[256];
__constant__	int gcBToCr[256];
__constant__	int gcYCBCRPixOff[256];				// used for BGR to YCBCR conversion
__constant__	unsigned short gcqtblY[64];
__constant__	unsigned short gcqtblCbCr[64];


////////////////////////////////////////////////////////////////////////////////

//	(1) Color convertion from rgb to ycbcr for one tile, 16*16 pixels;
//	(2) Y has 4 blocks, with block 0 from pY[0] to pY[63], 
//		block 1 from pY[64] to pY[127], block 2 from pY[128] to pY[191], ...
//	(3) With Cb/Cr subsampling, i.e. 2*2 pixels get one Cb and one Cr
//		IJG use average for better performance; we just pick one from four
//	(4) Do unsigned->signed conversion, i.e. substract 128 

//	This function processes a single pixel in each 16 x 16 block in a row
//	So, it is scheduled 64 times per set of 16 rows.

__global__ void kernelRGBToYCbCr( int *pOutput, int nXBlocks, int nW, int nH)
{ 
	int	x = threadIdx.x;			// pixel x in block
	int	y = threadIdx.y + blockIdx.x * BLOCK16;			// pixel y in block
	uchar4	val;

	int *pOut = pOutput + nXBlocks * YCBCRBLOCK * blockIdx.x;	// this the current block's output space

#pragma unroll
	for (int i = 0; i < nXBlocks ; i++, x += BLOCK16, pOut += YCBCRBLOCK)
	{
		if (x >= nW)
			x = nW - 1;
		if (y >= nH)
			y = nH - 1;
	
		val = tex2D( gpTexRGB, (float)x, (float)(y));	

		pOut[gcYCBCRPixOff[threadIdx.x + threadIdx.y * BLOCK16]] = 
				((gcRToY[ val.x ]  + gcGToY[ val.y ]  + gcBToY[ val.z ] )>>16) -128;	

		//	Equal to: (( x%2 == 0 )&&( y%2 == 0 ))
		if( (!(threadIdx.x & 1)) && (!(threadIdx.y & 1)) )
		{
			pOut[CBOFFSET + threadIdx.x / 2 + threadIdx.y * (BLOCK16 / 4)] = 
				((gcRToCb[ val.x ] + gcGToCb[ val.y ] + gcBToCb[ val.z ])>>16) -128;
			pOut[CROFFSET + threadIdx.x / 2 + threadIdx.y * (BLOCK16 / 4)] = 
				((gcRToCr[ val.x ] + gcGToCr[ val.y ] + gcBToCr[ val.z ])>>16) -128;
		}
	}
}


////////////////////////////////////////////////////////////////////////////
//	define some macroes 
	
//	Scale up the float with 1<<8; so (int)(0.382683433 * 1<<8 ) = 98

#define FIX_0_382683433  ((int)98)		/* FIX(0.382683433) */
#define FIX_0_541196100  ((int)139)		/* FIX(0.541196100) */
#define FIX_0_707106781  ((int)181)		/* FIX(0.707106781) */
#define FIX_1_306562965  ((int)334)		/* FIX(1.306562965) */
	
//	This macro changes float multiply into int multiply and right-shift
//	MULTIPLY(a, FIX_0_707106781) = (short)( 0.707106781 * a )
#define MULTIPLY(var,cons)  (int)(((cons) * (var)) >> 8 )

#define	USETRANSPOSE
#ifdef USETRANSPOSE

//	Being tricky with pOut here to ensure coalesced memory accesses - write out transpose

__global__ void kernelForwardDctRow(int *pOutBuf)	
{

	int	nInStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x * DCTSIZE;
	int	nOutStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x;
	int	*pOut = pOutBuf + nOutStart;

	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	int z1, z2, z3, z4, z5, z11, z13;
	
	tmp0 = tex1Dfetch( gpTexYCBCR, nInStart + 0) + tex1Dfetch( gpTexYCBCR, nInStart + 7);
	tmp7 = tex1Dfetch( gpTexYCBCR, nInStart + 0) - tex1Dfetch( gpTexYCBCR, nInStart + 7);
	tmp1 = tex1Dfetch( gpTexYCBCR, nInStart + 1) + tex1Dfetch( gpTexYCBCR, nInStart + 6);
	tmp6 = tex1Dfetch( gpTexYCBCR, nInStart + 1) - tex1Dfetch( gpTexYCBCR, nInStart + 6);
	tmp2 = tex1Dfetch( gpTexYCBCR, nInStart + 2) + tex1Dfetch( gpTexYCBCR, nInStart + 5);
	tmp5 = tex1Dfetch( gpTexYCBCR, nInStart + 2) - tex1Dfetch( gpTexYCBCR, nInStart + 5);
	tmp3 = tex1Dfetch( gpTexYCBCR, nInStart + 3) + tex1Dfetch( gpTexYCBCR, nInStart + 4);
	tmp4 = tex1Dfetch( gpTexYCBCR, nInStart + 3) - tex1Dfetch( gpTexYCBCR, nInStart + 4);
		
	/* Even part */
		
	tmp10 = tmp0 + tmp3;	/* phase 2 */
	tmp13 = tmp0 - tmp3;
	tmp11 = tmp1 + tmp2;
	tmp12 = tmp1 - tmp2;


	pOut[0 * DCTSIZE] = tmp10 + tmp11; /* phase 3 */
	pOut[4 * DCTSIZE] = tmp10 - tmp11;
		
	z1 = MULTIPLY(tmp12 + tmp13, FIX_0_707106781); /* c4 */
	pOut[2 * DCTSIZE] = tmp13 + z1;	/* phase 5 */
	pOut[6 * DCTSIZE] = tmp13 - z1;
	
	/* Odd part */
		
	tmp10 = tmp4 + tmp5;	/* phase 2 */
	tmp11 = tmp5 + tmp6;
	tmp12 = tmp6 + tmp7;
		
	/* The rotator is modified from fig 4-8 to avoid extra negations. */
	z5 = MULTIPLY(tmp10 - tmp12, FIX_0_382683433);	/* c6 */
	z2 = MULTIPLY(tmp10, FIX_0_541196100) + z5;		/* c2-c6 */
	z4 = MULTIPLY(tmp12, FIX_1_306562965) + z5;		/* c2+c6 */
	z3 = MULTIPLY(tmp11, FIX_0_707106781);			/* c4 */
		
	z11 = tmp7 + z3;		/* phase 5 */
	z13 = tmp7 - z3;
		
	pOut[5 * DCTSIZE] = z13 + z2;	/* phase 6 */
	pOut[3 * DCTSIZE] = z13 - z2;
	pOut[1 * DCTSIZE] = z11 + z4;
	pOut[7 * DCTSIZE] = z11 - z4;
}
#else
__global__ void kernelForwardDctRow(int *pOutBuf)	
{
	int	nInStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x * DCTSIZE;
	int *pOut = pOutBuf + nInStart;

	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	int z1, z2, z3, z4, z5, z11, z13;
	
	tmp0 = tex1Dfetch( gpTexYCBCR, nInStart + 0) + tex1Dfetch( gpTexYCBCR, nInStart + 7);
	tmp7 = tex1Dfetch( gpTexYCBCR, nInStart + 0) - tex1Dfetch( gpTexYCBCR, nInStart + 7);
	tmp1 = tex1Dfetch( gpTexYCBCR, nInStart + 1) + tex1Dfetch( gpTexYCBCR, nInStart + 6);
	tmp6 = tex1Dfetch( gpTexYCBCR, nInStart + 1) - tex1Dfetch( gpTexYCBCR, nInStart + 6);
	tmp2 = tex1Dfetch( gpTexYCBCR, nInStart + 2) + tex1Dfetch( gpTexYCBCR, nInStart + 5);
	tmp5 = tex1Dfetch( gpTexYCBCR, nInStart + 2) - tex1Dfetch( gpTexYCBCR, nInStart + 5);
	tmp3 = tex1Dfetch( gpTexYCBCR, nInStart + 3) + tex1Dfetch( gpTexYCBCR, nInStart + 4);
	tmp4 = tex1Dfetch( gpTexYCBCR, nInStart + 3) - tex1Dfetch( gpTexYCBCR, nInStart + 4);
		
	/* Even part */
		
	tmp10 = tmp0 + tmp3;	/* phase 2 */
	tmp13 = tmp0 - tmp3;
	tmp11 = tmp1 + tmp2;
	tmp12 = tmp1 - tmp2;


	pOut[0] = tmp10 + tmp11; /* phase 3 */
	pOut[4] = tmp10 - tmp11;
		
	z1 = MULTIPLY(tmp12 + tmp13, FIX_0_707106781); /* c4 */
	pOut[2] = tmp13 + z1;	/* phase 5 */
	pOut[6] = tmp13 - z1;
	
	/* Odd part */
		
	tmp10 = tmp4 + tmp5;	/* phase 2 */
	tmp11 = tmp5 + tmp6;
	tmp12 = tmp6 + tmp7;
		
	/* The rotator is modified from fig 4-8 to avoid extra negations. */
	z5 = MULTIPLY(tmp10 - tmp12, FIX_0_382683433);	/* c6 */
	z2 = MULTIPLY(tmp10, FIX_0_541196100) + z5;		/* c2-c6 */
	z4 = MULTIPLY(tmp12, FIX_1_306562965) + z5;		/* c2+c6 */
	z3 = MULTIPLY(tmp11, FIX_0_707106781);			/* c4 */
		
	z11 = tmp7 + z3;		/* phase 5 */
	z13 = tmp7 - z3;
		
	pOut[5] = z13 + z2;	/* phase 6 */
	pOut[3] = z13 - z2;
	pOut[1] = z11 + z4;
	pOut[7] = z11 - z4;
}
#endif

#ifdef	USETRANSPOSE
__global__ void kernelForwardDctCol(int *pOutBuf)	
{
	int	nInStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x * DCTSIZE;
	int	nOutStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x;
	int *pOut = pOutBuf + nOutStart;

	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	int z1, z2, z3, z4, z5, z11, z13;

	tmp0 = tex1Dfetch( gpTexYCBCR, nInStart + 0) + tex1Dfetch( gpTexYCBCR, nInStart + 7);
	tmp7 = tex1Dfetch( gpTexYCBCR, nInStart + 0) - tex1Dfetch( gpTexYCBCR, nInStart + 7);
	tmp1 = tex1Dfetch( gpTexYCBCR, nInStart + 1) + tex1Dfetch( gpTexYCBCR, nInStart + 6);
	tmp6 = tex1Dfetch( gpTexYCBCR, nInStart + 1) - tex1Dfetch( gpTexYCBCR, nInStart + 6);
	tmp2 = tex1Dfetch( gpTexYCBCR, nInStart + 2) + tex1Dfetch( gpTexYCBCR, nInStart + 5);
	tmp5 = tex1Dfetch( gpTexYCBCR, nInStart + 2) - tex1Dfetch( gpTexYCBCR, nInStart + 5);
	tmp3 = tex1Dfetch( gpTexYCBCR, nInStart + 3) + tex1Dfetch( gpTexYCBCR, nInStart + 4);
	tmp4 = tex1Dfetch( gpTexYCBCR, nInStart + 3) - tex1Dfetch( gpTexYCBCR, nInStart + 4);
		
	/* Even part */
		
	tmp10 = tmp0 + tmp3;	/* phase 2 */
	tmp13 = tmp0 - tmp3;
	tmp11 = tmp1 + tmp2;
	tmp12 = tmp1 - tmp2;
		
	pOut[DCTSIZE*0] = tmp10 + tmp11; /* phase 3 */
	pOut[DCTSIZE*4] = tmp10 - tmp11;
		
	z1 = MULTIPLY(tmp12 + tmp13, FIX_0_707106781); /* c4 */
	pOut[DCTSIZE*2] = tmp13 + z1; /* phase 5 */
	pOut[DCTSIZE*6] = tmp13 - z1;
		
	/* Odd part */
		
	tmp10 = tmp4 + tmp5;	/* phase 2 */
	tmp11 = tmp5 + tmp6;
	tmp12 = tmp6 + tmp7;
		
	/* The rotator is modified from fig 4-8 to avoid extra negations. */
	z5 = MULTIPLY(tmp10 - tmp12, FIX_0_382683433); /* c6 */
	z2 = MULTIPLY(tmp10, FIX_0_541196100) + z5; /* c2-c6 */
	z4 = MULTIPLY(tmp12, FIX_1_306562965) + z5; /* c2+c6 */
	z3 = MULTIPLY(tmp11, FIX_0_707106781); /* c4 */
	
	z11 = tmp7 + z3;		/* phase 5 */
	z13 = tmp7 - z3;
		
	pOut[DCTSIZE*5] = z13 + z2; /* phase 6 */
	pOut[DCTSIZE*3] = z13 - z2;
	pOut[DCTSIZE*1] = z11 + z4;
	pOut[DCTSIZE*7] = z11 - z4;
}

#else
__global__ void kernelForwardDctCol(int *pInBuf, int *pOutBuf)	
{
	int	nInStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x;
	int *pOut = pOutBuf + nInStart;
	int *pIn = pInBuf + nInStart;

	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	int z1, z2, z3, z4, z5, z11, z13;

	tmp0 = pIn[DCTSIZE*0] + pIn[DCTSIZE*7];
	tmp7 = pIn[DCTSIZE*0] - pIn[DCTSIZE*7];
	tmp1 = pIn[DCTSIZE*1] + pIn[DCTSIZE*6];
	tmp6 = pIn[DCTSIZE*1] - pIn[DCTSIZE*6];
	tmp2 = pIn[DCTSIZE*2] + pIn[DCTSIZE*5];
	tmp5 = pIn[DCTSIZE*2] - pIn[DCTSIZE*5];
	tmp3 = pIn[DCTSIZE*3] + pIn[DCTSIZE*4];
	tmp4 = pIn[DCTSIZE*3] - pIn[DCTSIZE*4];
		
	/* Even part */
		
	tmp10 = tmp0 + tmp3;	/* phase 2 */
	tmp13 = tmp0 - tmp3;
	tmp11 = tmp1 + tmp2;
	tmp12 = tmp1 - tmp2;
		
	pOut[DCTSIZE*0] = tmp10 + tmp11; /* phase 3 */
	pOut[DCTSIZE*4] = tmp10 - tmp11;
		
	z1 = MULTIPLY(tmp12 + tmp13, FIX_0_707106781); /* c4 */
	pOut[DCTSIZE*2] = tmp13 + z1; /* phase 5 */
	pOut[DCTSIZE*6] = tmp13 - z1;
		
	/* Odd part */
		
	tmp10 = tmp4 + tmp5;	/* phase 2 */
	tmp11 = tmp5 + tmp6;
	tmp12 = tmp6 + tmp7;
		
	/* The rotator is modified from fig 4-8 to avoid extra negations. */
	z5 = MULTIPLY(tmp10 - tmp12, FIX_0_382683433); /* c6 */
	z2 = MULTIPLY(tmp10, FIX_0_541196100) + z5; /* c2-c6 */
	z4 = MULTIPLY(tmp12, FIX_1_306562965) + z5; /* c2+c6 */
	z3 = MULTIPLY(tmp11, FIX_0_707106781); /* c4 */
	
	z11 = tmp7 + z3;		/* phase 5 */
	z13 = tmp7 - z3;
		
	pOut[DCTSIZE*5] = z13 + z2; /* phase 6 */
	pOut[DCTSIZE*3] = z13 - z2;
	pOut[DCTSIZE*1] = z11 + z4;
	pOut[DCTSIZE*7] = z11 - z4;
}
#endif

////////////////////////////////////////////////////////////////////////////////

__global__ void kernelQuantize(int * pInbuf, int *pOutBuf)
{
	int	nOffset = blockIdx.x * DCTBLOCKS * DCTSIZE * DCTSIZE + threadIdx.x;
	int	*pIn = pInbuf + nOffset;
	int *pOut = pOutBuf + nOffset;

	int temp;
	unsigned short qval;

	for (int i = 0; i < DCTBLOCKS; i++) 
	{
		if (i < 4)
			qval = gcqtblY[threadIdx.x];
		else
			qval = gcqtblCbCr[threadIdx.x];
		temp = pIn[i * DCTSIZE * DCTSIZE];
		
		/* Divide the coefficient value by qval, ensuring proper rounding.
		* Since C does not specify the direction of rounding for negative
		* quotients, we have to force the dividend positive for portability.
		*
		* In most files, at least half of the output values will be zero
		* (at default quantization settings, more like three-quarters...)
		* so we should ensure that this case is fast.  On many machines,
		* a comparison is enough cheaper than a divide to make a special test
		* a win.  Since both inputs will be nonnegative, we need only test
		* for a < b to discover whether a/b is 0.
		* If your machine's division is fast enough, define FAST_DIVIDE.
		*/

		// Notes: Actually we use the second expression !!
/*
#ifdef FAST_DIVIDE
#define DIVIDE_BY(a,b)	a /= b
#else
*/
#define DIVIDE_BY(a,b)	if (a >= b) a /= b; else a = 0
//#endif		
		
		if ( temp < 0) 
		{
			temp = -temp;
			temp += qval>>1;	/* for rounding */
			DIVIDE_BY(temp, qval);
			temp = -temp;
		} 
		else 
		{
			temp += qval>>1;	/* for rounding */
			DIVIDE_BY(temp, qval);
		}
		
		pOut[i * DCTSIZE * DCTSIZE] = temp;		
    }
}



////////////////////////////////////////////////////////////////////////////////

	/*
	* jpeg_natural_order[i] is the natural-order position of the i'th element
	* of zigzag order.
	*
	* When reading corrupted data, the Huffman decoders could attempt
	* to reference an entry beyond the end of this array (if the decoded
	* zero run length reaches past the end of the block).  To prevent
	* wild stores without adding an inner-loop test, we put some extra
	* "63"s after the real entries.  This will cause the extra coefficient
	* to be stored in location 63 of the block, not somewhere random.
	* The worst case would be a run-length of 15, which means we need 16
	* fake entries.
	*/
	static int jpeg_natural_order[64+16] = {
			 0,  1,  8, 16,  9,  2,  3, 10,
			17, 24, 32, 25, 18, 11,  4,  5,
			12, 19, 26, 33, 40, 48, 41, 34,
			27, 20, 13,  6,  7, 14, 21, 28,
			35, 42, 49, 56, 57, 50, 43, 36,
			29, 22, 15, 23, 30, 37, 44, 51,
			58, 59, 52, 45, 38, 31, 39, 46,
			53, 60, 61, 54, 47, 55, 62, 63,
			63, 63, 63, 63, 63, 63, 63, 63,//extra entries for safety
			63, 63, 63, 63, 63, 63, 63, 63
	};

//	Derived data constructed for each Huffman table 	
typedef struct tag_HUFFMAN_TABLE {
		unsigned int	code[256];	// code for each symbol 
		char			size[256];	// length of code for each symbol 
		//If no code has been allocated for a symbol S, size[S] is 0 

		/* These two fields directly represent the contents of a JPEG DHT marker */
		unsigned char bits[17];		/* bits[k] = # of symbols with codes of */
		/* length k bits; bits[0] is unused */
		unsigned char huffval[256];		/* The symbols, in order of incr code length */
								/* This field is used only during compression.  It's initialized false when
								* the table is created, and set true when it's been output to the file.
								* You could suppress output of a table by setting this to true.
								* (See jpeg_suppress_tables for an example.)*/
}HUFFMAN_TABLE;

	////////////////////////////////////////////////////////////////////////////
	//	Following data members should be computed in initialization

	unsigned short gnJPEGEncoderQuality, gnJPEGEncoderScale;




	//	used for write jpeg header
	unsigned char m_dqtY[64], m_dqtCbCr[64];

	HUFFMAN_TABLE m_htblEncYDC, m_htblEncYAC, m_htblEncCbCrDC, m_htblEncCbCrAC;

	////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////
	//	Following are should be initialized for compressing every image

	unsigned short gnJPEGEncoderWidth, gnJPEGEncoderHeight;

	//	Three dc records, used for dc differentize for Y/Cb/Cr
	int gnJPEGEncoderdcY, gnJPEGEncoderdcCb, gnJPEGEncoderdcCr;

	//	The size (in bits) and value (in 4 byte buffer) to be written out
	int m_nPutBits, m_nPutVal;

	unsigned char *m_pOutBuf;


////////////////////////////////////////////////////////////////////////////////
//JPEG marker codes 
typedef enum {		
  M_SOF0  = 0xc0,
  M_SOF1  = 0xc1,
  M_SOF2  = 0xc2,
  M_SOF3  = 0xc3,
  
  M_SOF5  = 0xc5,
  M_SOF6  = 0xc6,
  M_SOF7  = 0xc7,
  
  M_JPG   = 0xc8,
  M_SOF9  = 0xc9,
  M_SOF10 = 0xca,
  M_SOF11 = 0xcb,
  
  M_SOF13 = 0xcd,
  M_SOF14 = 0xce,
  M_SOF15 = 0xcf,
  
  M_DHT   = 0xc4,
  
  M_DAC   = 0xcc,
  
  M_RST0  = 0xd0,
  M_RST1  = 0xd1,
  M_RST2  = 0xd2,
  M_RST3  = 0xd3,
  M_RST4  = 0xd4,
  M_RST5  = 0xd5,
  M_RST6  = 0xd6,
  M_RST7  = 0xd7,
  
  M_SOI   = 0xd8,
  M_EOI   = 0xd9,
  M_SOS   = 0xda,
  M_DQT   = 0xdb,
  M_DNL   = 0xdc,
  M_DRI   = 0xdd,
  M_DHP   = 0xde,
  M_EXP   = 0xdf,
  
  M_APP0  = 0xe0,
  M_APP1  = 0xe1,
  M_APP2  = 0xe2,
  M_APP3  = 0xe3,
  M_APP4  = 0xe4,
  M_APP5  = 0xe5,
  M_APP6  = 0xe6,
  M_APP7  = 0xe7,
  M_APP8  = 0xe8,
  M_APP9  = 0xe9,
  M_APP10 = 0xea,
  M_APP11 = 0xeb,
  M_APP12 = 0xec,
  M_APP13 = 0xed,
  M_APP14 = 0xee,
  M_APP15 = 0xef,
  
  M_JPG0  = 0xf0,
  M_JPG13 = 0xfd,
  M_COM   = 0xfe,
  
  M_TEM   = 0x01,
  
  M_ERROR = 0x100
} JPEG_MARKER;

	// These are the sample quantization tables given in JPEG spec section K.1.
	// The spec says that the values given produce "good" quality, and
	// when divided by 2, "very good" quality.	
	
	static unsigned char std_luminance_quant_tbl[64] = 
	{
			16,  11,  10,  16,  24,  40,  51,  61,
			12,  12,  14,  19,  26,  58,  60,  55,
			14,  13,  16,  24,  40,  57,  69,  56,
			14,  17,  22,  29,  51,  87,  80,  62,
			18,  22,  37,  56,  68, 109, 103,  77,
			24,  35,  55,  64,  81, 104, 113,  92,
			49,  64,  78,  87, 103, 121, 120, 101,
			72,  92,  95,  98, 112, 100, 103,  99
	};
	static unsigned char std_chrominance_quant_tbl[64] = 
	{
			17,  18,  24,  47,  99,  99,  99,  99,
			18,  21,  26,  66,  99,  99,  99,  99,
			24,  26,  56,  99,  99,  99,  99,  99,
			47,  66,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99
	};





////////////////////////////////////////////////////////////////////////////////

#define emit_byte(val)	*m_pOutBuf++=(unsigned char)(val);

#define emit_2bytes(val)			\
*m_pOutBuf=(unsigned char)(((val)>>8)&0xFF);\
*(m_pOutBuf+1)=(unsigned char)((val)&0xFF);\
m_pOutBuf+=2;

#define emit_marker(val)			\
*m_pOutBuf=0xFF;\
*(m_pOutBuf+1)=(unsigned char)(val);\
m_pOutBuf+=2;
	



////////////////////////////////////////////////////////////////////////////////
//	Name:	_RTGPUJPEGEncoderInitColorTable()
//  Purpose:	
//			Save RGB->YCC colorspace conversion for reuse, only computing once
//			so dont need multiply in color conversion later

/* Notes:
 * 
 * YCbCr is defined per CCIR 601-1, except that Cb and Cr are
 * normalized to the range 0 .. 255 rather than -0.5 .. 0.5.
 * The conversion equations to be implemented are therefore
 *
 *	Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *	Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
 *	Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128
 *
 * (These numbers are derived from TIFF 6.0 section 21, dated 3-June-92.)
 * To avoid floating-point arithmetic, we represent the fractional constants
 * as integers scaled up by 2^16 (about 4 digits precision); we have to divide
 * the products by 2^16, with appropriate rounding, to get the correct answer.
 */

void _RTGPUJPEGEncoderInitColorTable( void )
{
	int i;
	int nScale	= 1L << 16;		//equal to power(2,16)
	int CBCR_OFFSET = 128<<16;

	int m_RToY[256],	m_GToY[256],	m_BToY[256];
	int m_RToCb[256],	m_GToCb[256],	m_BToCb[256];
	int m_RToCr[256],	m_GToCr[256],	m_BToCr[256];

	/*	
	*	nHalf is for (y, cb, cr) rounding, equal to (1L<<16)*0.5
	*	If (R,G,B)=(0,0,1), then Cb = 128.5, should round to 129
	*	Using these tables will produce 129 too: 
	*	Cb	= (int)((RToCb[0] + GToCb[0] + BToCb[1]) >> 16)
	*		= (int)(( 0 + 0 + 1L<<15 + 1L<<15 + 128 * 1L<<16 ) >> 16)
	*		= (int)(( 1L<<16 + 128 * 1L<<16 ) >> 16 )
	*		= 129
	*/
	int nHalf = nScale >> 1;	

	for( i=0; i<256; i++ )
	{
		m_RToY[ i ]	= (int)( 0.29900 * nScale + 0.5 ) * i;
		m_GToY[ i ]	= (int)( 0.58700 * nScale + 0.5 ) * i;
		m_BToY[ i ]	= (int)( 0.11400 * nScale + 0.5 ) * i + nHalf;

		m_RToCb[ i ] = (int)( 0.16874 * nScale + 0.5 ) * (-i);
		m_GToCb[ i ] = (int)( 0.33126 * nScale + 0.5 ) * (-i);
		m_BToCb[ i ] = (int)( 0.50000 * nScale + 0.5 ) * i + 
										CBCR_OFFSET + nHalf - 1;

		m_RToCr[ i ] = m_BToCb[ i ];
		m_GToCr[ i ] = (int)( 0.41869 * nScale + 0.5 ) * (-i);
		m_BToCr[ i ] = (int)( 0.08131 * nScale + 0.5 ) * (-i);
	}
	RTGPUSafeCall(cudaMemcpyToSymbol(gcRToY, m_RToY, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcGToY, m_GToY, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcBToY, m_BToY, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcRToCb, m_RToCb, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcGToCb, m_GToCb, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcBToCb, m_BToCb, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcRToCr, m_RToCr, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcGToCr, m_GToCr, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcBToCr, m_BToCr, 256 * sizeof(int)));

	//	Do YCBCR offset table 

	int x, y, py[4], pYCBCRPixOff[256];

	for( x = 0; x < 4; x++ )
		py[ x ] = 64 * x;

	for( y=0; y<16; y++ )
	{
		for( x=0; x<16; x++ )
		{
			pYCBCRPixOff[y * 16 + x] = py[((y>>3)<<1) + (x>>3)] ++;
		}
	}
	RTGPUSafeCall(cudaMemcpyToSymbol(gcYCBCRPixOff, pYCBCRPixOff, 256 * sizeof(int)));
}

////////////////////////////////////////////////////////////////////////////////
void ScaleTable(unsigned char* tbl, int scale, int max)
{
	int i, temp, half = max/2;

	for (i = 0; i < 64; i++) 
	{
		// (1) user scale up
		temp = (int)(( gnJPEGEncoderScale * tbl[i] + half ) / max );

		// limit to baseline range 
		if (temp <= 0)
			temp = 1;
		if (temp > 255)
			temp = 255;

		// (2) scaling needed for AA&N algorithm
		tbl[i] = (unsigned char)temp;
	}
}

////////////////////////////////////////////////////////////////////////////////

void ScaleQuantTable(
			unsigned short* tblRst,		//result quant table
			unsigned char* tblStd,		//standard quant table
			unsigned short* tblAan		//scale factor for AAN dct
			)
{
	int i, temp, half = 1<<10;
	for (i = 0; i < 64; i++) 
	{
		// (1) user scale up
		temp = (int)(( gnJPEGEncoderScale * tblStd[i] + 50 ) / 100 );

		// limit to baseline range 
		if (temp <= 0) 
			temp = 1;
		if (temp > 255)
			temp = 255;		

		// (2) scaling needed for AA&N algorithm
		tblRst[i] = (unsigned short)(( temp * tblAan[i] + half )>>11 );
	}
}


////////////////////////////////////////////////////////////////////////////////
//	_RTGPUJPEGEncoderInitQuantTable will produce customized quantization table into:
//		m_tblYQuant[0..63] and m_tblCbCrQuant[0..63]

void _RTGPUJPEGEncoderInitQuantTable( void )
{

	unsigned short qtblY[64], qtblCbCr[64];

	/*  For AA&N IDCT method, divisors are equal to quantization
	*	coefficients scaled by scalefactor[row]*scalefactor[col], where
	*		scalefactor[0] = 1
	*		scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
	*	We apply a further scale factor of 8.
	*/	
	static unsigned short aanscales[64] = {
			/* precomputed values scaled up by 14 bits */
			16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
			22725, 31521, 29692, 26722, 22725, 17855, 12299,  6270,
			21407, 29692, 27969, 25172, 21407, 16819, 11585,  5906,
			19266, 26722, 25172, 22654, 19266, 15137, 10426,  5315,
			16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
			12873, 17855, 16819, 15137, 12873, 10114,  6967,  3552,
			 8867, 12299, 11585, 10426,  8867,  6967,  4799,  2446,
			 4520,  6270,  5906,  5315,  4520,  3552,  2446,  1247
	};
	
	// Safety checking. Convert 0 to 1 to avoid zero divide. 
	gnJPEGEncoderScale = gnJPEGEncoderQuality;

	if (gnJPEGEncoderScale <= 0) 
		gnJPEGEncoderScale = 1;
	if (gnJPEGEncoderScale > 100) 
		gnJPEGEncoderScale = 100;
	
	//	Non-linear map: 1->5000, 10->500, 25->200, 50->100, 75->50, 100->0
	if (gnJPEGEncoderScale < 50)
		gnJPEGEncoderScale = 5000 / gnJPEGEncoderScale;
	else
		gnJPEGEncoderScale = 200 - gnJPEGEncoderScale*2;

	// use std to initialize
	memcpy( m_dqtY,		std_luminance_quant_tbl,	64 );
	memcpy( m_dqtCbCr,	std_chrominance_quant_tbl,	64 );

	//	scale dqt for writing jpeg header
	ScaleTable( m_dqtY,		gnJPEGEncoderScale, 100 );
	ScaleTable( m_dqtCbCr,	gnJPEGEncoderScale, 100 );		

	//	Scale the Y and CbCr quant table, respectively
	ScaleQuantTable( qtblY,    &std_luminance_quant_tbl[0],   aanscales );
	ScaleQuantTable( qtblCbCr, &std_chrominance_quant_tbl[0], aanscales );
	RTGPUSafeCall(cudaMemcpyToSymbol(gcqtblY, qtblY, 64 * sizeof(unsigned short)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcqtblCbCr, qtblCbCr, 64 * sizeof(unsigned short)));
}


////////////////////////////////////////////////////////////////////////////////

//	Compute the derived values for a Huffman table.	
//	also, add bits[] and huffval[] to Hufftable for writing jpeg file header

void ComputeHuffmanTable(
		unsigned char *	pBits, 
		unsigned char * pVal,
		HUFFMAN_TABLE * pTbl	)
{
	int p, i, l, lastp, si;
	char huffsize[257];
	unsigned int huffcode[257];
	unsigned int code;

	// First we copy bits and huffval
	memcpy( pTbl->bits,		pBits,	sizeof(pTbl->bits) );
	memcpy( pTbl->huffval,  pVal,	sizeof(pTbl->huffval) );
	
	/* Figure C.1: make table of Huffman code length for each symbol */
	/* Note that this is in code-length order. */
	
	p = 0;
	for (l = 1; l <= 16; l++) {
		for (i = 1; i <= (int) pBits[l]; i++)
			huffsize[p++] = (char) l;
	}
	huffsize[p] = 0;
	lastp = p;
	
	/* Figure C.2: generate the codes themselves */
	/* Note that this is in code-length order. */
	
	code = 0;
	si = huffsize[0];
	p = 0;
	while (huffsize[p]) {
		while (((int) huffsize[p]) == si) {
			huffcode[p++] = code;
			code++;
		}
		code <<= 1;
		si++;
	}
	
	/* Figure C.3: generate encoding tables */
	/* These are code and size indexed by symbol value */
	
	/* Set any codeless symbols to have code length 0;
	* this allows EmitBits to detect any attempt to emit such symbols.
	*/
	memset( pTbl->size, 0, sizeof( pTbl->size ) );
	
	for (p = 0; p < lastp; p++) {
		pTbl->code[ pVal[p] ] = huffcode[p];
		pTbl->size[ pVal[p] ] = huffsize[p];
	}
}



////////////////////////////////////////////////////////////////////////////////
//	Prepare four Huffman tables:
//		HUFFMAN_TABLE m_htblYDC, m_htblYAC, m_htblCbCrDC, m_htblCbCrAC;

void _RTGPUJPEGEncoderInitHuffmanTable( void )
{
	//	Y dc component
	static unsigned char bitsYDC[17] =
    { 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
	static unsigned char valYDC[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
	

	//	CbCr dc
	static unsigned char bitsCbCrDC[17] =
    { 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
	static unsigned char valCbCrDC[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
	

	//	Y ac component
	static unsigned char bitsYAC[17] =
    { 0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
	static unsigned char valYAC[] =
    { 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
	0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
	0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
	0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
	0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
	0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
	0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
	0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
	0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
	0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
	0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
	0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
	0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
	0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
	0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
	0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
	0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
	0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
	0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
	0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa };
	

	//	CbCr ac
	static unsigned char bitsCbCrAC[17] =
    { 0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
	static unsigned char valCbCrAC[] =
    { 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
	0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
	0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
	0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
	0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
	0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
	0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
	0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
	0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
	0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
	0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
	0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
	0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
	0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
	0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
	0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
	0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
	0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa };

	//	Compute four derived Huffman tables
	ComputeHuffmanTable( bitsYDC, valYDC, &m_htblEncYDC );
	ComputeHuffmanTable( bitsYAC, valYAC, &m_htblEncYAC );

	ComputeHuffmanTable( bitsCbCrDC, valCbCrDC, &m_htblEncCbCrDC );
	ComputeHuffmanTable( bitsCbCrAC, valCbCrAC, &m_htblEncCbCrAC );
}

////////////////////////////////////////////////////////////////////////////////

/* Outputting bits to the file */

/* Only the right 24 bits of put_buffer are used; the valid bits are
 * left-justified in this part.  At most 16 bits can be passed to EmitBits
 * in one call, and we never retain more than 7 bits in put_buffer
 * between calls, so 24 bits are sufficient.
 */

inline bool EmitBits(
		unsigned int code,		//Huffman code
		int size				//Size in bits of the Huffman code
		)
{
	/* This routine is heavily used, so it's worth coding tightly. */
	int put_buffer = (int) code;
	int put_bits = m_nPutBits;
	
	/* if size is 0, caller used an invalid Huffman table entry */
	if (size == 0)
		return false;
	
	put_buffer &= (((int)1)<<size) - 1; /* mask off any extra bits in code */
	
	put_bits += size;					/* new number of bits in buffer */
	
	put_buffer <<= 24 - put_bits;		/* align incoming bits */
	
	put_buffer |= m_nPutVal;			/* and merge with old buffer contents */
	
	//	If there are more than 8 bits, write it out
	unsigned char uc;
	while (put_bits >= 8) 
	{
		//	Write one byte out !!!!
		uc = (unsigned char) ((put_buffer >> 16) & 0xFF);
		emit_byte(uc);
	
		if (uc == 0xFF) {		//need to stuff a zero byte?
			emit_byte(0);	//	Write one byte out !!!!
		}

		put_buffer <<= 8;
		put_bits -= 8;
	}
	
	m_nPutVal	= put_buffer; /* update state variables */
	m_nPutBits	= put_bits;
	
	return true;
}

////////////////////////////////////////////////////////////////////////////////

inline void EmitLeftBits(void)
{
	if (! EmitBits(0x7F, 7)) /* fill 7 bits with ones */
		return;
/*	
	unsigned char uc = (unsigned char) ((m_nPutVal >> 16) & 0xFF);
	emit_byte(uc);		//	Write one byte out !!!!
*/	
	m_nPutVal  = 0;
	m_nPutBits = 0;
}


///////////////////////////////////////////////////////////////////////////////

void write_soi()
{
	emit_marker(M_SOI);
}

void write_app0()
{
  /*
   * Length of APP0 block	(2 bytes)
   * Block ID			(4 bytes - ASCII "JFIF")
   * Zero byte			(1 byte to terminate the ID string)
   * Version Major, Minor	(2 bytes - 0x01, 0x01)
   * Units			(1 byte - 0x00 = none, 0x01 = inch, 0x02 = cm)
   * Xdpu			(2 bytes - dots per unit horizontal)
   * Ydpu			(2 bytes - dots per unit vertical)
   * Thumbnail X size		(1 byte)
   * Thumbnail Y size		(1 byte)
   */
  
  emit_marker(M_APP0);
  
  emit_2bytes(2 + 4 + 1 + 2 + 1 + 2 + 2 + 1 + 1); /* length */

  emit_byte(0x4A);	/* Identifier: ASCII "JFIF" */
  emit_byte(0x46);
  emit_byte(0x49);
  emit_byte(0x46);
  emit_byte(0);

  /* We currently emit version code 1.01 since we use no 1.02 features.
   * This may avoid complaints from some older decoders.
   */
  emit_byte(1);		/* Major version */
  emit_byte(1);		/* Minor version */
  emit_byte(1); /* Pixel size information */
  emit_2bytes(300);
  emit_2bytes(300);
  emit_byte(0);		/* No thumbnail image */
  emit_byte(0);
}

void write_dqt(int index)//0:Y;1:CbCr
{
	unsigned char* dqt;
	if( index == 0 )
		dqt = &m_dqtY[0];//changed from std with quality
	else
		dqt = &m_dqtCbCr[0];

	//only allow prec = 0;

	emit_marker(M_DQT);
	emit_2bytes(67);//length
	emit_byte(index);

	int i;
	unsigned char qval;
	for (i = 0; i < 64; i++) 
	{
        qval = (unsigned char) (dqt[jpeg_natural_order[i]]);
		emit_byte(qval);
    }
}


//currently support M_SOF0 baseline implementation
void write_sof(int code)
{
	emit_marker(code);
	emit_2bytes(17); //length

	emit_byte(8);//cinfo->data_precision);
	emit_2bytes(gnJPEGEncoderHeight);
	emit_2bytes(gnJPEGEncoderWidth);
	emit_byte(3);//cinfo->num_components);

	//for Y
	emit_byte(1);//compptr->component_id);
	emit_byte(34);//(compptr->h_samp_factor << 4) + compptr->v_samp_factor);
	emit_byte(0);//quant_tbl_no

	//for Cb
	emit_byte(2);//compptr->component_id);
	emit_byte(17);//(compptr->h_samp_factor << 4) + compptr->v_samp_factor);
	emit_byte(1);//quant_tbl_no

	//for Cr
	emit_byte(3);//compptr->component_id);
	emit_byte(17);//(compptr->h_samp_factor << 4) + compptr->v_samp_factor);
	emit_byte(1);//quant_tbl_no
}

void write_dht(int IsCbCr, int IsAc)
{
	HUFFMAN_TABLE *htbl;
	int index;
	if( IsCbCr )
	{
		if( IsAc )
		{
			htbl = &m_htblEncCbCrAC;
			index = 17;
		}
		else
		{
			htbl = &m_htblEncCbCrDC;
			index = 1;
		}
	}
	else
	{
		if( IsAc )
		{
			htbl = &m_htblEncYAC;
			index = 16;
		}
		else
		{
			htbl = &m_htblEncYDC;
			index = 0;
		}
	}

	emit_marker(M_DHT);

	int i, length = 0;
    for (i = 1; i <= 16; i++)
		length += htbl->bits[i];
	
	emit_2bytes(length + 2 + 1 + 16);
	
    emit_byte(index);

	for (i = 1; i <= 16; i++)
		emit_byte(htbl->bits[i]);
    
    for (i = 0; i < length; i++)//varible-length
		emit_byte(htbl->huffval[i]);    
}

void write_sos()
{
	emit_marker(M_SOS);

	int length = 2 * 3 + 2 + 1 + 3;
	emit_2bytes(length);

	emit_byte(3);//cinfo->comps_in_scan

	//Y
	emit_byte(1);//index
	emit_byte(0);//dc and ac tbl use 0-th tbl

	//Cb
	emit_byte(2);//index
	emit_byte(0x11);//dc and ac tbl use 1-th tbl

	//Cr
	emit_byte(3);//index
	emit_byte(0x11);//dc and ac tbl use 1-th tbl

	emit_byte(0);//Ss
	emit_byte(0x3F);//Se
	emit_byte(0);//  Ah/Al
}
////////////////////////////////////////////////////////////////////////////////

//write soi, app0, Y_dqt, CbCr_dqt, sof, 4 * dht, sos.
void WriteJpegHeader(void)
{
	write_soi();

	write_app0();

	write_dqt(0);//Y

	write_dqt(1);//cbcr

	write_sof(M_SOF0);

	write_dht(0, 0);//m_htblYDC
	write_dht(0, 1);//m_htblYAC
	write_dht(1, 0);//m_htblCbCrDC
	write_dht(1, 1);//m_htblCbCrAC

	write_sos();
}



////////////////////////////////////////////////////////////////////////////////


 



////////////////////////////////////////////////////////////////////////////////

 bool HuffmanEncode( 
		int* pCoef,				//	DCT coefficients
		int iBlock				//	0,1,2,3:Y; 4:Cb; 5:Cr;
		)
{	

	
	int temp, temp2, nbits, k, r, i;
	int *block = pCoef;
	int *pLastDc = &gnJPEGEncoderdcY;
	HUFFMAN_TABLE *dctbl, *actbl;

	if( iBlock < 4 )
	{
		dctbl = & m_htblEncYDC;
		actbl = & m_htblEncYAC;
//		pLastDc = &gnJPEGEncoderdcY;	
	}
	else
	{
		dctbl = & m_htblEncCbCrDC;
		actbl = & m_htblEncCbCrAC;

		if( iBlock == 4 )
			pLastDc = &gnJPEGEncoderdcCb;
		else
			pLastDc = &gnJPEGEncoderdcCr;
	}
	
	/* Encode the DC coefficient difference per section F.1.2.1 */
	
	temp = temp2 = block[0] - (*pLastDc);
	*pLastDc = block[0];
	
	if (temp < 0) {
		temp = -temp;		/* temp is abs value of input */
		/* For a negative input, want temp2 = bitwise complement of abs(input) */
		/* This code assumes we are on a two's complement machine */
		temp2 --;
	}
	
	/* Find the number of bits needed for the magnitude of the coefficient */
	nbits = 0;
	while (temp) {
		nbits ++;
		temp >>= 1;
	}
	
	//	Write category number
	if (! EmitBits( dctbl->code[nbits], dctbl->size[nbits] ))
		return false;

	//	Write category offset
	if (nbits)			/* EmitBits rejects calls with size 0 */
	{
		if (! EmitBits( (unsigned int) temp2, nbits ))
			return false;
	}
	
	////////////////////////////////////////////////////////////////////////////
	/* Encode the AC coefficients per section F.1.2.2 */
	
	r = 0;			/* r = run length of zeros */
	
	for (k = 1; k < 64; k++) 
	{
		if ((temp = block[jpeg_natural_order[k]]) == 0) 
		{
			r++;
		} 
		else 
		{
			/* if run length > 15, must emit special run-length-16 codes (0xF0) */
			while (r > 15) {
				if (! EmitBits( actbl->code[0xF0], actbl->size[0xF0] ))
					return false;
				r -= 16;
			}
			
			temp2 = temp;
			if (temp < 0) {
				temp = -temp;		/* temp is abs value of input */
				/* This code assumes we are on a two's complement machine */
				temp2--;
			}
			
			/* Find the number of bits needed for the magnitude of the coefficient */
			nbits = 1;		/* there must be at least one 1 bit */
			while ((temp >>= 1))
				nbits++;
			
			/* Emit Huffman symbol for run length / number of bits */
			i = (r << 4) + nbits;
			if (! EmitBits( actbl->code[i], actbl->size[i] ))
				return false;
						
			//	Write Category offset
			if (! EmitBits( (unsigned int) temp2, nbits ))
				return false;
						
			r = 0;
		}
	}
	
	//If all the left coefs were zero, emit an end-of-block code
	if (r > 0)
	{
		if (! EmitBits( actbl->code[0], actbl->size[0] ))
			return false;
	}		
	
	return true;
}


////////////////////////////////////////////////////////////////////////////////
//	function Purpose:	compress one 16*16 pixels with jpeg
//	destination is m_pOutBuf, in jpg format

bool CompressOneTile(int *pCoef)
{
	//	The DCT outputs are returned scaled up by a factor of 8;
	//	they therefore have a range of +-8K for 8-bit source data 

	//	Do Y/Cb/Cr components, Y: 4 blocks; Cb: 1 block; Cr: 1 block
	int i;
	for( i=0; i<6; i++ )
	{
		HuffmanEncode( pCoef + i*64, i );//output into m_pOutBuf	
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//	Prepare for all the tables needed, 
//	eg. quantization tables, huff tables, color convert tables
//	1 <= nQuality <= 100, is used for quantization scaling
//	Computing once, and reuse them again and again !!!!!!!

extern "C" void _RTGPUJPEGInitEncoder( )
{
	gnJPEGEncoderQuality = 80;

	//	prepare color convert table, from bgr to ycbcr
	_RTGPUJPEGEncoderInitColorTable( );

	//	prepare two quant tables, one for Y, and another for CbCr
	_RTGPUJPEGEncoderInitQuantTable( );

	//	prepare four huffman tables: 
	_RTGPUJPEGEncoderInitHuffmanTable( );
}


extern "C" bool _RTGPUJPEGCompress(int srcSlot, unsigned char *outBuf, int& outputBytes)
{
	int nW, nH;
	RTGPU_IMAGE	*SI;
    cudaChannelFormatDesc desc;

	int *pGPUYSIZE1;										// GPU buffer for YCbCr/DCT data
	int *pGPUYSIZE2;										// another one
	int *pQuant;											// buffer for quantized data

	int		nYCbCrSize;										// total size of YCbCr data

	int xTile, yTile, cxTile, cyTile;

	RTGPU_SLOTPTR(srcSlot, SI);
	nW = SI->width;
	nH = SI->height;
	
	RTGPUTrace("RTGPUJPEGCompress");

	assert(SI->color);

	gnJPEGEncoderWidth = nW;
	gnJPEGEncoderHeight = nH;
	m_pOutBuf = outBuf;

	//write soi, app0, Y_dqt, CbCr_dqt, sof, 4 * dht, sos.
	WriteJpegHeader();
		
	//	horizontal and vertical count of tile, macroblocks, 
	//	or MCU(Minimum Coded Unit), in 16*16 pixels
	cxTile = (nW + 15) >> 4;	
	cyTile = (nH + 15) >> 4;

	nYCbCrSize = cxTile * cyTile * YCBCRBLOCK * sizeof(int);
	RTGPUSafeCall(cudaMalloc(&(pGPUYSIZE1), nYCbCrSize));
	RTGPUSafeCall(cudaMalloc(&(pGPUYSIZE2), nYCbCrSize));

	desc = cudaCreateChannelDesc<uchar4>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, gpTexRGB, SI->image, desc, nW, nH, nW * sizeof(uchar4)));

	dim3	blocksConv(cxTile, cyTile);
	dim3	threadsConv(BLOCK16, BLOCK16);

	kernelRGBToYCbCr<<<cyTile, threadsConv>>>(pGPUYSIZE1, cxTile, nW, nH); // Get YCbCr data

	RTGPUSafeCall(cudaUnbindTexture(gpTexRGB));

	desc = cudaCreateChannelDesc<int>();
	RTGPUSafeCall(cudaBindTexture(NULL, gpTexYCBCR, pGPUYSIZE1, desc, nYCbCrSize));
	pQuant = (int *)malloc(nYCbCrSize);

	dim3	threadsDCT(DCTSIZE, DCTBLOCKS);		// .x = 8, .y = 6

	kernelForwardDctRow<<<cxTile * cyTile, threadsDCT>>>(pGPUYSIZE2);

#ifdef	USETRANSPOSE
	RTGPUSafeCall(cudaBindTexture(NULL, gpTexYCBCR, pGPUYSIZE2, desc, nYCbCrSize));

	kernelForwardDctCol<<<cxTile * cyTile, threadsDCT>>>(pGPUYSIZE1);

	RTGPUSafeCall(cudaUnbindTexture(gpTexYCBCR));
#else
	kernelForwardDctCol<<<cxTile * cyTile, threadsDCT>>>(pGPUYSIZE2, pGPUYSIZE1);
#endif

	kernelQuantize<<<cxTile * cyTile, DCTSIZE * DCTSIZE>>>(pGPUYSIZE1, pGPUYSIZE2);

	RTGPUSafeCall(cudaThreadSynchronize());
	RTGPUSafeCall(cudaMemcpy(pQuant, pGPUYSIZE2, nYCbCrSize, cudaMemcpyDeviceToHost));
	RTGPUSafeCall(cudaFree(pGPUYSIZE1));
	RTGPUSafeCall(cudaFree(pGPUYSIZE2));

	//	three dc values set to zero, needed for compressing one new image
	gnJPEGEncoderdcY = gnJPEGEncoderdcCb = gnJPEGEncoderdcCr = 0;

	//	Initialize size (in bits) and value to be written out
	m_nPutBits = 0;
	m_nPutVal = 0;

	//	Run all the tiles, or macroblocks, or MCUs

	for( yTile = 0; yTile < cyTile; yTile++ )
	{
		for( xTile = 0; xTile < cxTile; xTile++ )
		{
			//	Compress this full tile with jpeg algorithm here !!!!!
			//	The compressed data length for this tile is return by nTileBytes
			if( ! CompressOneTile(pQuant + (yTile * cxTile + xTile) * YCBCRBLOCK))
				return false;			
		}
	}

	free(pQuant);
	
	//	Maybe there are some bits left, send them here
	if( m_nPutBits > 0 )
	{
		EmitLeftBits( );
	}

	// write EOI; end of Image
	emit_marker(M_EOI);
	outputBytes = m_pOutBuf - outBuf;
	return true;
}


////////////////////////////////////////////////////////////////////////////////

