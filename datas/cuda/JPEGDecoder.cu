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
#define	CBOFFSET	256				// offset to Cb part
#define	CROFFSET	320				// offset to Cr part

typedef struct {
  int component_id;		/* identifier for this component (0..255) */
  int component_index;		/* its index in SOF or cinfo->comp_info[] */
  int h_samp_factor;		/* horizontal sampling factor (1..4) */
  int v_samp_factor;		/* vertical sampling factor (1..4) */
  int quant_tbl_no;		/* quantization table selector (0..3) */
} jpeg_component_info;


texture<int>				gpTexIDCT;				// the IDCT texture
texture<unsigned char>		gpTexYCBCR;				// the YCBCR texture

__constant__	int gcYCBCRPixOff[256];				// used for YCBCR tp BGR conversion
__constant__	unsigned char gctblRange[5*256+128];
__constant__	int	gcCrToR[256];
__constant__	int	gcCrToG[256];
__constant__	int	gcCbToB[256];
__constant__	int	gcCbToG[256];
__constant__	unsigned short gcqtblY[64];
__constant__	unsigned short gcqtblCbCr[64];



// Derived data constructed for each Huffman table 
typedef struct{
	int				mincode[17];	// smallest code of length k 
	int				maxcode[18];	// largest code of length k (-1 if none) 
	int				valptr[17];		// huffval[] index of 1st symbol of length k
	unsigned char	bits[17];		// bits[k] = # of symbols with codes of 
	unsigned char	huffval[256];	// The symbols, in order of incr code length 
	int				look_nbits[256];// # bits, or 0 if too long
	unsigned char	look_sym[256];	// symbol, or unused
} HUFFTABLE;


	////////////////////////////////////////////////////////////////////////////
	//	Following are initialized when create a new decoder

	unsigned short gnJPEGDecoderQuality, gnJPEGDecoderScale;

	unsigned char m_tblRange[5*256+128];
	
	//	To speed up, we save YCbCr=>RGB color map tables
	int m_CrToR[256], m_CrToG[256],	m_CbToB[256], m_CbToG[256];

	//	To speed up, we precompute two DCT quant tables
	unsigned short m_qtblY[64], m_qtblCbCr[64];

	HUFFTABLE m_htblDecYDC, m_htblDecYAC, m_htblDecCbCrDC, m_htblDecCbCrAC;


////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////
	//	Following data are initialized for decoding every image

	unsigned short gnJPEGDecoderWidth, gnJPEGDecoderHeight, m_nMcuSize, m_nBlocksInMcu;

	int gnJPEGDecoderdcY, gnJPEGDecoderdcCb, gnJPEGDecoderdcCr;

	int m_nGetBits, m_nGetBuff, m_nDataBytesLeft;

	unsigned char * m_pData;

	int m_nPrecision, m_nComponent;

	int restart_interval, restarts_to_go, unread_marker, next_restart_num;

	jpeg_component_info comp_info[3];

	////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//	Color conversion and up-sampling
//	if m_nBlocksInMcu==3, no need to up-sampling. McuSize is blockDim.x and blockDim.y

__global__ void kernelYCbCrToRGB( uchar4 *pOutput, int nXBlocks, int nW, int nH)
{ 
	uchar4	val;
	int		nCbOff, nCrOff;
	unsigned char	y, cb, cr;
	int		nYCbCrBlock;
	int		nRangeLimit = 256;

	uchar4 *pOut = pOutput + nW * blockDim.y * blockIdx.x;	// this the current block's output space

	if (blockDim.x == 16)
	{
		nCbOff = 256;
		nCrOff = 320;
		nYCbCrBlock = 384;
	}
	else
	{
		nCbOff = 64;
		nCrOff = 128;
		nYCbCrBlock = 192;
	}
	int	nInOff = nXBlocks * nYCbCrBlock * blockIdx.x;

	#pragma unroll
	for (int i = 0; i < nXBlocks ; i++, pOut += blockDim.x, nInOff += nYCbCrBlock)
	{
		y = tex1Dfetch( gpTexYCBCR, nInOff + gcYCBCRPixOff[threadIdx.x + threadIdx.y * blockDim.x]);
		cb = tex1Dfetch( gpTexYCBCR, nInOff + nCbOff + threadIdx.x / 2 + (threadIdx.y / 2) * 8);
		cr = tex1Dfetch( gpTexYCBCR, nInOff + nCrOff + threadIdx.x / 2 + (threadIdx.y / 2) * 8);

		//	Blue
		val.z = gctblRange[ nRangeLimit + y + gcCbToB[cb] ];

		//	Green
		val.y = gctblRange[ nRangeLimit + y + ((gcCbToG[cb] + gcCrToG[cr])>>16) ];

		//	Red
		val.x = gctblRange[ nRangeLimit + y + gcCrToR[cr] ];

		pOut[threadIdx.x + threadIdx.y * nW] = val;
	}
}

////////////////////////////////////////////////////////////////////////////////

//	AA&N DCT algorithm implemention

#define RANGE_MASK	1023 //2 bits wider than legal samples
#define PASS1_BITS  2
#define IDESCALE(x,n)  ((int) ((x)>>n) )
#define FIX_1_082392200  ((int)277)		/* FIX(1.082392200) */
#define FIX_1_414213562  ((int)362)		/* FIX(1.414213562) */
#define FIX_1_847759065  ((int)473)		/* FIX(1.847759065) */
#define FIX_2_613125930  ((int)669)		/* FIX(2.613125930) */
	
#define MULTIPLY(var,cons)  ((int) ((var)*(cons))>>8 )

__global__ void kernelInverseDctCol(int *pOutBuf)
{

	int	nInStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x;
	int *pOut = pOutBuf + nInStart;

	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	int z5, z10, z11, z12, z13;

	unsigned short* quantptr;	
	int		dcval;

	if( blockIdx.y < 4 )
		quantptr = gcqtblY + threadIdx.x;
	else
		quantptr = gcqtblCbCr + threadIdx.x;
	
	//Pass 1: process columns from input (inptr), store into work array(wsptr)
	
    /* Due to quantization, we will usually find that many of the input
	* coefficients are zero, especially the AC terms.  We can exploit this
	* by short-circuiting the IDCT calculation for any column in which all
	* the AC terms are zero.  In that case each output is equal to the
	* DC coefficient (with scale factor as needed).
	* With typical images and quantization tables, half or more of the
	* column DCT calculations can be simplified this way.
	*/
		
	if ((	tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 1) | 
			tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 2) | 
			tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 3) |
			tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 4) | 
			tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 5) | 
			tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 6) |
			tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 7)) == 0) 
	{
		/* AC terms all zero */
		dcval = (int)( tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 0) * quantptr[DCTSIZE*0] );
			
		pOut[DCTSIZE*0] = dcval;
		pOut[DCTSIZE*1] = dcval;
		pOut[DCTSIZE*2] = dcval;
		pOut[DCTSIZE*3] = dcval;
		pOut[DCTSIZE*4] = dcval;
		pOut[DCTSIZE*5] = dcval;
		pOut[DCTSIZE*6] = dcval;
		pOut[DCTSIZE*7] = dcval;
		return;
	}
		
	/* Even part */
		
	tmp0 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 0) * quantptr[DCTSIZE*0];
	tmp1 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 2) * quantptr[DCTSIZE*2];
	tmp2 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 4) * quantptr[DCTSIZE*4];
	tmp3 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 6) * quantptr[DCTSIZE*6];
		
	tmp10 = tmp0 + tmp2;	/* phase 3 */
	tmp11 = tmp0 - tmp2;
		
	tmp13 = tmp1 + tmp3;	/* phases 5-3 */
	tmp12 = MULTIPLY(tmp1 - tmp3, FIX_1_414213562) - tmp13; /* 2*c4 */
		
	tmp0 = tmp10 + tmp13;	/* phase 2 */
	tmp3 = tmp10 - tmp13;
	tmp1 = tmp11 + tmp12;
	tmp2 = tmp11 - tmp12;
		
	/* Odd part */
		
	tmp4 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 1) * quantptr[DCTSIZE*1];
	tmp5 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 3) * quantptr[DCTSIZE*3];
	tmp6 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 5) * quantptr[DCTSIZE*5];
	tmp7 = tex1Dfetch( gpTexIDCT, nInStart + DCTSIZE * 7) * quantptr[DCTSIZE*7];
		
	z13 = tmp6 + tmp5;		/* phase 6 */
	z10 = tmp6 - tmp5;
	z11 = tmp4 + tmp7;
	z12 = tmp4 - tmp7;
		
	tmp7  = z11 + z13;		/* phase 5 */
	tmp11 = MULTIPLY(z11 - z13, FIX_1_414213562); /* 2*c4 */
	
	z5	  = MULTIPLY(z10 + z12, FIX_1_847759065); /* 2*c2 */
	tmp10 = MULTIPLY(z12, FIX_1_082392200) - z5; /* 2*(c2-c6) */
	tmp12 = MULTIPLY(z10, - FIX_2_613125930) + z5; /* -2*(c2+c6) */
		
	tmp6 = tmp12 - tmp7;	/* phase 2 */
	tmp5 = tmp11 - tmp6;
	tmp4 = tmp10 + tmp5;
		
	pOut[DCTSIZE*0] = (int) (tmp0 + tmp7);
	pOut[DCTSIZE*7] = (int) (tmp0 - tmp7);
	pOut[DCTSIZE*1] = (int) (tmp1 + tmp6);
	pOut[DCTSIZE*6] = (int) (tmp1 - tmp6);
	pOut[DCTSIZE*2] = (int) (tmp2 + tmp5);
	pOut[DCTSIZE*5] = (int) (tmp2 - tmp5);
	pOut[DCTSIZE*4] = (int) (tmp3 + tmp4);
	pOut[DCTSIZE*3] = (int) (tmp3 - tmp4);
}


__global__ void kernelInverseDctRow(unsigned char *pOutBuf)	
{
	int	nInStart = (blockIdx.x * DCTBLOCKS + threadIdx.y) * DCTSIZE * DCTSIZE + threadIdx.x * DCTSIZE;
	unsigned char *pOut = pOutBuf + nInStart;

	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	int z5, z10, z11, z12, z13;

	int		nRangeLimit = 384;
	int		dcval;

	/* Pass 2: process rows from work array, store into output array. */
	/* Note that we must descale the results by a factor of 8 == 2**3, */
	/* and also undo the PASS1_BITS scaling. */

	/* Rows of zeroes can be exploited in the same way as we did with columns.
	* However, the column calculation has created many nonzero AC terms, so
	* the simplification applies less often (typically 5% to 10% of the time).
	* On machines with very fast multiplication, it's possible that the
	* test takes more time than it's worth.  In that case this section
	* may be commented out.
	*/
		
	if ((	tex1Dfetch( gpTexIDCT, nInStart + 1) | 
			tex1Dfetch( gpTexIDCT, nInStart + 2) | 
			tex1Dfetch( gpTexIDCT, nInStart + 3) | 
			tex1Dfetch( gpTexIDCT, nInStart + 4) | 
			tex1Dfetch( gpTexIDCT, nInStart + 5) | 
			tex1Dfetch( gpTexIDCT, nInStart + 6) |
			tex1Dfetch( gpTexIDCT, nInStart + 7)) == 0) 
	{
		/* AC terms all zero */
		dcval = (int) gctblRange[nRangeLimit + (tex1Dfetch( gpTexIDCT, nInStart + 0) >> 5) & RANGE_MASK];		
		pOut[0] = dcval;
		pOut[1] = dcval;
		pOut[2] = dcval;
		pOut[3] = dcval;
		pOut[4] = dcval;
		pOut[5] = dcval;
		pOut[6] = dcval;
		pOut[7] = dcval;
		return;
	}
		
	/* Even part */
		
	tmp10 = ((int) tex1Dfetch( gpTexIDCT, nInStart + 0) + (int) tex1Dfetch( gpTexIDCT, nInStart + 4));
	tmp11 = ((int) tex1Dfetch( gpTexIDCT, nInStart + 0) - (int) tex1Dfetch( gpTexIDCT, nInStart + 4));
		
	tmp13 = ((int) tex1Dfetch( gpTexIDCT, nInStart + 2) + (int) tex1Dfetch( gpTexIDCT, nInStart + 6));
	tmp12 = MULTIPLY((int) tex1Dfetch( gpTexIDCT, nInStart + 2) - (int) tex1Dfetch( gpTexIDCT, nInStart + 6), FIX_1_414213562)
			- tmp13;
		
	tmp0 = tmp10 + tmp13;
	tmp3 = tmp10 - tmp13;
	tmp1 = tmp11 + tmp12;
	tmp2 = tmp11 - tmp12;
		
	/* Odd part */
		
	z13 = (int) tex1Dfetch( gpTexIDCT, nInStart + 5) + (int) tex1Dfetch( gpTexIDCT, nInStart + 3);
	z10 = (int) tex1Dfetch( gpTexIDCT, nInStart + 5) - (int) tex1Dfetch( gpTexIDCT, nInStart + 3);
	z11 = (int) tex1Dfetch( gpTexIDCT, nInStart + 1) + (int) tex1Dfetch( gpTexIDCT, nInStart + 7);
	z12 = (int) tex1Dfetch( gpTexIDCT, nInStart + 1) - (int) tex1Dfetch( gpTexIDCT, nInStart + 7);
		
	tmp7 = z11 + z13;		/* phase 5 */
	tmp11 = MULTIPLY(z11 - z13, FIX_1_414213562); /* 2*c4 */
		
	z5    = MULTIPLY(z10 + z12, FIX_1_847759065); /* 2*c2 */
	tmp10 = MULTIPLY(z12, FIX_1_082392200) - z5; /* 2*(c2-c6) */
	tmp12 = MULTIPLY(z10, - FIX_2_613125930) + z5; /* -2*(c2+c6) */
	
	tmp6 = tmp12 - tmp7;	/* phase 2 */
	tmp5 = tmp11 - tmp6;
	tmp4 = tmp10 + tmp5;
		
	/* Final output stage: scale down by a factor of 8 and range-limit */
		
	pOut[0] = gctblRange[nRangeLimit + IDESCALE(tmp0 + tmp7, PASS1_BITS+3) & RANGE_MASK];
	pOut[7] = gctblRange[nRangeLimit + IDESCALE(tmp0 - tmp7, PASS1_BITS+3) & RANGE_MASK];
	pOut[1] = gctblRange[nRangeLimit + IDESCALE(tmp1 + tmp6, PASS1_BITS+3) & RANGE_MASK];
	pOut[6] = gctblRange[nRangeLimit + IDESCALE(tmp1 - tmp6, PASS1_BITS+3) & RANGE_MASK];
	pOut[2] = gctblRange[nRangeLimit + IDESCALE(tmp2 + tmp5, PASS1_BITS+3) & RANGE_MASK];
	pOut[5] = gctblRange[nRangeLimit + IDESCALE(tmp2 - tmp5, PASS1_BITS+3) & RANGE_MASK];
	pOut[4] = gctblRange[nRangeLimit + IDESCALE(tmp3 + tmp4, PASS1_BITS+3) & RANGE_MASK];
	pOut[3] = gctblRange[nRangeLimit + IDESCALE(tmp3 - tmp4, PASS1_BITS+3) & RANGE_MASK];
}

////////////////////////////////////////////////////////////////////////////////

typedef enum {			/* JPEG marker codes */
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


/*
* jpeg_natural_order[i] is the natural-order position of the i'th 
* element of zigzag order.
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
	static const int jpeg_natural_order[64+16] = {
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

#define INPUT_2BYTES(src)  (unsigned short)(((*src)<<8)+(*(src+1)));src+=2;

#define INPUT_BYTE(src)	(unsigned char)(*src++)



////////////////////////////////////////////////////////////////////////////////

// read exact marker, two bytes, no stuffing allowed
int ReadOneMarker(void)
{
	if( INPUT_BYTE(m_pData) != 255 )
		return -1;
	int marker = INPUT_BYTE(m_pData);
	return marker;
}

////////////////////////////////////////////////////////////////////////////////

// Skip over an unknown or uninteresting variable-length marker
void SkipMarker(void)
{
  int length = (int)INPUT_2BYTES(m_pData);

	// Just skip; who care what info is? 
	m_pData += length - 2;
}


////////////////////////////////////////////////////////////////////////////////
void GetDqt( void )
{
	int length;
	length = (int)INPUT_2BYTES(m_pData);
	length -= 2;
	
	unsigned short *qtb;
	int n, i;	
	while (length > 0) 
	{
		n = INPUT_BYTE(m_pData);
		length --;
		n &= 0x0F;//dqt for Y, or Cb/Cr?


		if( n == 0 )
			qtb = m_qtblY;
		else
			qtb = m_qtblCbCr;
		
		for (i = 0; i < 64; i++) 
		{
			qtb[jpeg_natural_order[i]] = INPUT_BYTE(m_pData);
		}
		length -= 64;
	}
}

////////////////////////////////////////////////////////////////////////////////

// get width and height, and component info

void get_sof (bool is_prog, bool is_arith)
{
	int length = (int)INPUT_2BYTES(m_pData);

	m_nPrecision = (int)INPUT_BYTE(m_pData);//1 byte

	gnJPEGDecoderHeight = (unsigned short)INPUT_2BYTES(m_pData);

	gnJPEGDecoderWidth = (unsigned short)INPUT_2BYTES(m_pData);

	m_nComponent = (int)INPUT_BYTE(m_pData);//1 byte
	
	length -= 8;

	jpeg_component_info *compptr;
	compptr = comp_info;
	int ci, c;
	for (ci = 0; ci < m_nComponent; ci++) 
	{
		compptr->component_index = ci;
	
		compptr->component_id = (int)INPUT_BYTE(m_pData);//1 byte

		c = (int)INPUT_BYTE(m_pData);//1 byte
		compptr->h_samp_factor = (c >> 4) & 15;
		compptr->v_samp_factor = (c     ) & 15;
		
		if(( ci==0 )&&( c!=34 ))
		{
			char info[100];
			sprintf(info, "comp 0 samp_factor = %d", c );
			RTGPUTrace(info);
		}
		
		compptr->quant_tbl_no = (int)INPUT_BYTE(m_pData);//1 byte
		
		compptr++;
	}

	if(	( comp_info[0].h_samp_factor == 1 )&&
		( comp_info[0].v_samp_factor == 1 ))
	{
		m_nMcuSize		= 8;
		m_nBlocksInMcu	= 3;
	}
	else
	{
		m_nMcuSize		= 16;//default
		m_nBlocksInMcu	= 6;
	}
}

///////////////////////////////////////////////////////////////////////////////

void get_dht(void)
{
	int length = (int)INPUT_2BYTES(m_pData);
	length -= 2;
	while(length>0)
	{
		//0:dc_huff_tbl[0]; 16:ac_huff_tbl[0];
		//1:dc_huff_tbl[1]; 17:ac_huff_tbl[1]
		int index = INPUT_BYTE(m_pData);

		// decide which table to receive data
		HUFFTABLE* htblptr;
		switch(index){
		case 0:
			htblptr = &m_htblDecYDC;
			break;
		case 16:
			htblptr = &m_htblDecYAC;
			break;
		case 1:
			htblptr = &m_htblDecCbCrDC;
			break;
		case 17:
			htblptr = &m_htblDecCbCrAC;
			break;
		}
	
		int count, i;
		//
		// read in bits[]
		//
		htblptr->bits[0] = 0;
		count = 0;
		for (i = 1; i <= 16; i++) {
			htblptr->bits[i] = INPUT_BYTE(m_pData);
			count += htblptr->bits[i];
		}		
		length -= (1 + 16);
		
		//
		// read in huffval
		//
		for (i = 0; i < count; i++){
			htblptr->huffval[i] = INPUT_BYTE(m_pData);
		}
		length -= count;
	}
}

///////////////////////////////////////////////////////////////////////////////

void get_sos(void)
{
	int length = (int)INPUT_2BYTES(m_pData);
	int n = INPUT_BYTE(m_pData);// Number of components

	// Collect the component-spec parameters
	int i, c;
	for (i = 0; i < n; i++) 
	{
		c = INPUT_BYTE(m_pData);
		c = INPUT_BYTE(m_pData);
		
	}
	
	// Collect the additional scan parameters Ss, Se, Ah/Al.
//	int Ss, Se, Ah, Al;
//	Ss	= INPUT_BYTE(m_pData);
//	Se	= INPUT_BYTE(m_pData);
//	c	= INPUT_BYTE(m_pData);
//	Ah = (c >> 4) & 15;
//	Al = (c     ) & 15;
	c	= INPUT_BYTE(m_pData);
	c	= INPUT_BYTE(m_pData);
	c	= INPUT_BYTE(m_pData);

	next_restart_num = 0;
}

////////////////////////////////////////////////////////////////////////////////

void get_dri()
{
	int length = (int)INPUT_2BYTES(m_pData);
	restart_interval = INPUT_2BYTES(m_pData);

	restarts_to_go = restart_interval; 

	char info[100];
	sprintf(info, "restart_interval=%d", restart_interval );
	RTGPUTrace(info);
}



////////////////////////////////////////////////////////////////////////////////

//return: -1, error; 0, SOS, start of scan; 1: EOI, end of image

int read_markers (
	unsigned char *pInBuf,	//in, source data, in jpg format
	int cbInBuf,			//in, count bytes for in buffer
	int& nWidth,			//out, image width in pixels
	int& nHeight,			//out, image height
	int& nHeadSize			//out, header size in bytes
	)
{
	m_pData = pInBuf;
	int retval = -1;
	for (;;) 
	{
		// IJG use first_marker() and next_marker()
		int marker = ReadOneMarker();

		// read more info according to the marker
		// the order of cases is in jpg file made by ms paint
		switch (marker) 
		{
		case M_SOI:
//			if (! get_soi(cinfo))
//				return -1;//JPEG_SUSPENDED;
			break;

		case M_APP0:
		case M_APP1:
		case M_APP2:
		case M_APP3:
		case M_APP4:
		case M_APP5:
		case M_APP6:
		case M_APP7:
		case M_APP8:
		case M_APP9:
		case M_APP10:
		case M_APP11:
		case M_APP12:
		case M_APP13:
		case M_APP14:
		case M_APP15:
			SkipMarker();//JFIF APP0 marker, or Adobe APP14 marker
			break;

		case M_DQT:// maybe twice, one for Y, another for Cb/Cr
			GetDqt();
			break;
			
		case M_SOF0:		//* Baseline
		case M_SOF1:		//* Extended sequential, Huffman 
			get_sof(false, false);
			break;
			
		case M_SOF2:		//* Progressive, Huffman 
			//get_sof(true, false);	
			RTGPUTrace("Prog + Huff is not supported");
			return -1;
			
		case M_SOF9:		//* Extended sequential, arithmetic 
			//get_sof(false, true);
			RTGPUTrace("sequential + Arith is not supported");
			return -1;
			
		case M_SOF10:		//* Progressive, arithmetic 
			//get_sof(true, true);
			RTGPUTrace("Prog + Arith is not supported");
			return -1;
					
		case M_DHT:
			get_dht();//4 tables: dc/ac * Y/CbCr
			break;

		case M_SOS://Start of Scan
			get_sos();
			retval = 0;//JPEG_REACHED_SOS;
			
			nWidth = gnJPEGDecoderWidth;
			nHeight = gnJPEGDecoderHeight;
			nHeadSize = m_pData - pInBuf;
			return retval;

		//the following marker are not needed for jpg made by ms paint
		case M_COM:
			SkipMarker();
			break;

		case M_DRI:
			get_dri();
			break;
			
			
/*			
		Currently unsupported SOFn types 
		case M_SOF3:		 Lossless, Huffman
		case M_SOF5:		 Differential sequential, Huffman 
		case M_SOF6:		 Differential progressive, Huffman 
		case M_SOF7:		 Differential lossless, Huffman 
		case M_JPG:			 Reserved for JPEG extensions 
		case M_SOF11:		 Lossless, arithmetic 
		case M_SOF13:		 Differential sequential, arithmetic 
		case M_SOF14:		 Differential progressive, arithmetic
		case M_SOF15:		 Differential lossless, arithmetic 
			return -1;//ERREXIT1(cinfo, JERR_SOF_UNSUPPORTED, cinfo->unread_marker);
			break;
			
		case M_EOI:
			TRACEMS(cinfo, 1, JTRC_EOI);
			cinfo->unread_marker = 0;	
			return 1;//JPEG_REACHED_EOI;
			
		case M_DAC:
			if (! get_dac(cinfo))
				return -1;//JPEG_SUSPENDED;
			break;

		case M_RST0:		
		case M_RST1:
		case M_RST2:
		case M_RST3:
		case M_RST4:
		case M_RST5:
		case M_RST6:
		case M_RST7:
		case M_TEM:
			break;
			
		case M_DNL:			
		if (! skip_variable(cinfo))
				return -1;//JPEG_SUSPENDED;
			break;
*/			
		default:			
			/* must be DHP, EXP, JPGn, or RESn */
			/* For now, we treat the reserved markers as fatal errors since they are
			* likely to be used to signal incompatible JPEG Part 3 extensions.
			* Once the JPEG 3 version-number marker is well defined, this code
			* ought to change!
			*/
			return -1;//	ERREXIT1(cinfo, JERR_UNKNOWN_MARKER, cinfo->unread_marker);
		}
		/* Successfully processed marker, so reset state variable */
		unread_marker = 0;
	}
}


///////////////////////////////////////////////////////////////////////////////

void read_restart_marker (void)
{
  /* Obtain a marker unless we already did. */
  /* Note that next_marker will complain if it skips any data. */
  if (unread_marker == 0) 
  {
    unread_marker = ReadOneMarker();
  }

  if (unread_marker == ((int) M_RST0 + next_restart_num)) 
  {
	  /* Normal case --- swallow the marker and let entropy decoder continue */
	  unread_marker = 0;
  } 
  else {
    /* Uh-oh, the restart markers have been messed up. */
    /* Let the data source manager determine how to resync. */
	  
	//lin changed:
	/*
    if (! (*cinfo->src->resync_to_restart) (cinfo,
					    cinfo->marker->next_restart_num))
      return FALSE;
	  */
  }

  /* Update next-restart state */
  next_restart_num = (next_restart_num + 1) & 7;

  return;
}





////////////////////////////////////////////////////////////////////////////////
//	prepare_range_limit_table(): Set m_tblRange[5*256+128 = 1408]
//	range table is used for range limiting of idct results
/*	On most machines, particularly CPUs with pipelines or instruction prefetch,
 *	a (subscript-check-less) C table lookup
 *			x = sample_range_limit[x];
 *	is faster than explicit tests
 *			if (x < 0)  x = 0;
 *			else if (x > MAXJSAMPLE)  x = MAXJSAMPLE;
 */

void SetRangeTable( void )
{
	unsigned char *tbl;

	//	m_tblRange[0, ..., 255], limit[x] = 0 for x < 0
	memset( m_tblRange, 0, 256 );

	//	m_tblRange[256, ..., 511], limit[x] = x
	tbl = m_tblRange + 256;
	for( int i=0; i<256; i++ )
		*(tbl++) = (unsigned char) i;

	// m_tblRange[512, ..., 895]: first half of post-IDCT table
	tbl = m_tblRange + 512;
	for (int i = 128; i < 512; i++)
		*(tbl++) = 255;

	//	m_tblRange[896, ..., 1280]: Second half of post-IDCT table
	memset( m_tblRange + 896, 0, 384);

	// [1280, 1407] = [256, 384]
	memcpy( m_tblRange + 1280, m_tblRange + 256, 128);

	RTGPUSafeCall(cudaMemcpyToSymbol(gctblRange, m_tblRange, 256 * 5 + 128));

}


////////////////////////////////////////////////////////////////////////////////

/**************** YCbCr -> RGB conversion: most common case **************/

/*
 * YCbCr is defined per CCIR 601-1, except that Cb and Cr are
 * normalized to the range 0..MAXJSAMPLE rather than -0.5 .. 0.5.
 * The conversion equations to be implemented are therefore
 *	R = Y                + 1.40200 * Cr
 *	G = Y - 0.34414 * Cb - 0.71414 * Cr
 *	B = Y + 1.77200 * Cb
 * where Cb and Cr represent the incoming values less CENTERJSAMPLE.
 * (These numbers are derived from TIFF 6.0 section 21, dated 3-June-92.)
 *
 * To avoid floating-point arithmetic, we represent the fractional constants
 * as integers scaled up by 2^16 (about 4 digits precision); we have to divide
 * the products by 2^16, with appropriate rounding, to get the correct answer.
 * Notice that Y, being an integral input, does not contribute any fraction
 * so it need not participate in the rounding.
 *
 * For even more speed, we avoid doing any multiplications in the inner loop
 * by precalculating the constants times Cb and Cr for all possible values.
 * For 8-bit JSAMPLEs this is very reasonable (only 256 entries per table);
 * for 12-bit samples it is still acceptable.  It's not very reasonable for
 * 16-bit samples, but if you want lossless storage you shouldn't be changing
 * colorspace anyway.
 * The Cr=>R and Cb=>B values can be rounded to integers in advance; the
 * values for the G calculation are left scaled up, since we must add them
 * together before rounding.
 */

void RTGPUJPEGDecoderInitColorTable( void )
{
	int i, x;
	int nScale	= 1L << 16;		//equal to power(2,16)
	int nHalf	= nScale >> 1;

#define FIX(x) ((int) ((x) * nScale + 0.5))

	/* i is the actual input pixel value, in the range 0..MAXJSAMPLE */
    /* The Cb or Cr value we are thinking of is x = i - CENTERJSAMPLE */
    /* Cr=>R value is nearest int to 1.40200 * x */
	/* Cb=>B value is nearest int to 1.77200 * x */
	/* Cr=>G value is scaled-up -0.71414 * x */
	/* Cb=>G value is scaled-up -0.34414 * x */
    /* We also add in ONE_HALF so that need not do it in inner loop */
	for (i = 0, x = -128; i < 256; i++, x++) 
	{
		m_CrToR[i] = (int) ( FIX(1.40200) * x + nHalf ) >> 16;
		m_CbToB[i] = (int) ( FIX(1.77200) * x + nHalf ) >> 16;
		m_CrToG[i] = (int) (- FIX(0.71414) * x );
		m_CbToG[i] = (int) (- FIX(0.34414) * x + nHalf );
	}

	//	Do YCBCR offset table 

	int y, py[4], pYCBCRPixOff[256];

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
	RTGPUSafeCall(cudaMemcpyToSymbol(gcCrToR, m_CrToR, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcCbToB, m_CbToB, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcCrToG, m_CrToG, 256 * sizeof(int)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcCbToG, m_CbToG, 256 * sizeof(int)));
}

////////////////////////////////////////////////////////////////////////////////

void ScaleQuantTable(
			unsigned short* tblRst,		//result quant table
			unsigned short* tblStd,		//standard quant table
			unsigned short* tblAan		//scale factor for AAN dct
			)
{
	int i, half = 1<<11;
	
	for (i = 0; i < 64; i++) 
	{
		// scaling needed for AA&N algorithm
		tblRst[i] = (unsigned short)(( tblStd[i] * tblAan[i] + half ) >> 12 );
	}
}

////////////////////////////////////////////////////////////////////////////////
//	RTGPUJPEGDecoderInitQuantTable will produce customized quantization table into:
//		m_tblYQuant[0..63] and m_tblCbCrQuant[0..63]

void RTGPUJPEGDecoderInitQuantTable( void )
{
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
	
	//	Scale the Y and CbCr quant table, respectively
	ScaleQuantTable( m_qtblY,	 m_qtblY,		aanscales );
	ScaleQuantTable( m_qtblCbCr, m_qtblCbCr,	aanscales );	
	RTGPUSafeCall(cudaMemcpyToSymbol(gcqtblY, m_qtblY, 64 * sizeof(short)));
	RTGPUSafeCall(cudaMemcpyToSymbol(gcqtblCbCr, m_qtblCbCr, 64 * sizeof(short)));

}

////////////////////////////////////////////////////////////////////////////////

//	Compute the derived values for a Huffman table.	

void ComputeHuffmanTable(HUFFTABLE * dtbl	)
{
	int p, i, l, si;
	int lookbits, ctr;
	char huffsize[257];
	unsigned int huffcode[257];
	unsigned int code;

	unsigned char *pBits = dtbl->bits;
	unsigned char *pVal  = dtbl->huffval;

	/* Figure C.1: make table of Huffman code length for each symbol */
	/* Note that this is in code-length order. */
	p = 0;
	for (l = 1; l <= 16; l++) {
		for (i = 1; i <= (int) pBits[l]; i++)
			huffsize[p++] = (char) l;
	}
	huffsize[p] = 0;
	
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
	
	/* Figure F.15: generate decoding tables for bit-sequential decoding */
	
	p = 0;
	for (l = 1; l <= 16; l++) {
		if (pBits[l]) {
			dtbl->valptr[l] = p; /* huffval[] index of 1st symbol of code length l */
			dtbl->mincode[l] = huffcode[p]; /* minimum code of length l */
			p += pBits[l];
			dtbl->maxcode[l] = huffcode[p-1]; /* maximum code of length l */
		} else {
			dtbl->maxcode[l] = -1;	/* -1 if no codes of this length */
		}
	}
	dtbl->maxcode[17] = 0xFFFFFL; /* ensures jpeg_huff_decode terminates */
	
	/* Compute lookahead tables to speed up decoding.
	 * First we set all the table entries to 0, indicating "too long";
	 * then we iterate through the Huffman codes that are short enough and
	 * fill in all the entries that correspond to bit sequences starting
	 * with that code.	 */
	
	memset( dtbl->look_nbits, 0, sizeof(int)*256 );
	
	int HUFF_LOOKAHEAD = 8;
	p = 0;
	for (l = 1; l <= HUFF_LOOKAHEAD; l++) 
	{
		for (i = 1; i <= (int) pBits[l]; i++, p++) 
		{
			/* l = current code's length, 
			p = its index in huffcode[] & huffval[]. Generate left-justified
			code followed by all possible bit sequences */
			lookbits = huffcode[p] << (HUFF_LOOKAHEAD-l);
			for (ctr = 1 << (HUFF_LOOKAHEAD-l); ctr > 0; ctr--) 
			{
				dtbl->look_nbits[lookbits] = l;
				dtbl->look_sym[lookbits] = pVal[p];
				lookbits++;
			}
		}
	}
}



////////////////////////////////////////////////////////////////////////////////
//	Prepare four Huffman tables:
//		HUFFMAN_TABLE m_htblYDC, m_htblYAC, m_htblCbCrDC, m_htblCbCrAC;

void RTGPUJPEGDecoderInitHuffmanTable( void )
{
	//	Using dht got from jpeg file header
	ComputeHuffmanTable( &m_htblDecYDC );
	ComputeHuffmanTable( &m_htblDecYAC );
	ComputeHuffmanTable( &m_htblDecCbCrDC );
	ComputeHuffmanTable( &m_htblDecCbCrAC );
}

////////////////////////////////////////////////////////////////////////////////

//	Below are difficult and complex HUFFMAN decoding !!!!!

////////////////////////////////////////////////////////////////////////////////

void FillBitBuffer( void )
{
	unsigned char uc;
	while( m_nGetBits < 25 )	//#define MIN_GET_BITS  (32-7)
	{
		if( m_nDataBytesLeft > 0 )//Are there some data?
		{ 
			/* Attempt to read a byte */
			if (unread_marker != 0)
				goto no_more_data;	/* can't advance past a marker */

			uc = *m_pData++;
			m_nDataBytesLeft --;			
			
			// If it's 0xFF, check and discard stuffed zero byte
			if (uc == 0xFF) 
			{
				do 
				{
					uc = *m_pData++;
					m_nDataBytesLeft --;
				}while (uc == 0xFF);
				
				if (uc == 0) 
				{
					// Found FF/00, which represents an FF data byte
					uc = 0xFF;
				} 
				else 
				{
					// Oops, it's actually a marker indicating end of compressed data.
					// Better put it back for use later 
					
					unread_marker = uc;

no_more_data:					
					// There should be enough bits still left in the data segment;
					// if so, just break out of the outer while loop.
					//if (m_nGetBits >= nbits)
					if (m_nGetBits >= 0)
						break;
				}
			}

			m_nGetBuff = (m_nGetBuff << 8) | ((int) uc);
			m_nGetBits += 8;			
		}
		else
			break;
	}
}

////////////////////////////////////////////////////////////////////////////////

inline int GetBits(int nbits) 
{
	if( m_nGetBits < nbits )//we should read nbits bits to get next data
		FillBitBuffer();
	m_nGetBits -= nbits;
	return (int) (m_nGetBuff >> m_nGetBits) & ((1<<nbits)-1);
}

////////////////////////////////////////////////////////////////////////////////
//	Special Huffman decode:
//	(1) For codes with length > 8
//	(2) For codes with length < 8 while data is finished

int SpecialDecode( HUFFTABLE* htbl, int nMinBits )
{
	
	int l = nMinBits;
	int code;
	
	/* HUFF_DECODE has determined that the code is at least min_bits */
	/* bits long, so fetch that many bits in one swoop. */

	code = GetBits(l);
	
	/* Collect the rest of the Huffman code one bit at a time. */
	/* This is per Figure F.16 in the JPEG spec. */
	while (code > htbl->maxcode[l]) {
		code <<= 1;
		code |= GetBits(1);
		l++;
	}
	
	/* With garbage input we may reach the sentinel value l = 17. */
	if (l > 16) {
		return 0;			/* fake a zero as the safest result */
	}
	
	return htbl->huffval[ htbl->valptr[l] +	(int)(code - htbl->mincode[l]) ];
}

////////////////////////////////////////////////////////////////////////////////
//	To find dc or ac value according to category and category offset

inline int ValueFromCategory(int nCate, int nOffset)
{
/*	//Method 1: 
	//On some machines, a shift and add will be faster than a table lookup.
	#define HUFF_EXTEND(x,s) \
	((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) : (x)) 
*/
	//Method 2: Table lookup
	
	//If (nOffset < half[nCate]), then value is below zero
	//Otherwise, value is above zero, and just the nOffset
	static const int half[16] =		/* entry n is 2**(n-1) */
	{ 0, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080,
    0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000 };
	
	//start[i] is the starting value in this category; surely it is below zero
	static const int start[16] =	/* entry n is (-1 << n) + 1 */
	{ 0, ((-1)<<1) + 1, ((-1)<<2) + 1, ((-1)<<3) + 1, ((-1)<<4) + 1,
    ((-1)<<5) + 1, ((-1)<<6) + 1, ((-1)<<7) + 1, ((-1)<<8) + 1,
    ((-1)<<9) + 1, ((-1)<<10) + 1, ((-1)<<11) + 1, ((-1)<<12) + 1,
    ((-1)<<13) + 1, ((-1)<<14) + 1, ((-1)<<15) + 1 };	

	return ( nOffset < half[nCate] ? nOffset + start[nCate] : nOffset);	
}


////////////////////////////////////////////////////////////////////////////////
//get category number for dc, or (0 run length, ac category) for ac

//	The max length for Huffman codes is 15 bits; so we use 32 bits buffer	
//	m_nGetBuff, with the validated length is m_nGetBits.
//	Usually, more than 95% of the Huffman codes will be 8 or fewer bits long
//	To speed up, we should pay more attention on the codes whose length <= 8

inline int GetCategory( HUFFTABLE* htbl )
{
	//	If left bits < 8, we should get more data
	if( m_nGetBits < 8 )
		FillBitBuffer( );

	//	Call special process if data finished; min bits is 1
	if( m_nGetBits < 8 )
		return SpecialDecode( htbl, 1 );

	//	Peek the first valid byte	
	int look = ((m_nGetBuff>>(m_nGetBits - 8))& 0xFF);
	int nb = htbl->look_nbits[look];

	if( nb ) 
	{ 
		m_nGetBits -= nb;
		return htbl->look_sym[look]; 
	} 
	else	//Decode long codes with length >= 9
		return SpecialDecode( htbl, 9 );
}


////////////////////////////////////////////////////////////////////////////////
//	HuffmanDecode( coef, i ); //source is m_pData; coef is result

void HuffmanDecode( 
		int* coef,//	out, DCT coefficients
		int iBlock	//	0,1,2,3:Y; 4:Cb; 5:Cr; or 0:Y;1:Cb;2:Cr
		)
{	
	int* pLastDC;
	int s, k, r;

	HUFFTABLE *dctbl, *actbl;

	if( iBlock < m_nBlocksInMcu - 2 )
	{
		dctbl = &m_htblDecYDC;
		actbl = &m_htblDecYAC;
		pLastDC = &gnJPEGDecoderdcY;
	}
	else
	{
		dctbl = &m_htblDecCbCrDC;
		actbl = &m_htblDecCbCrAC;
		if( iBlock == m_nBlocksInMcu - 2 )
			pLastDC = &gnJPEGDecoderdcCb;
		else
			pLastDC = &gnJPEGDecoderdcCr;
	}

	memset( coef, 0, sizeof(int) * 64 );
	
    /* Section F.2.2.1: decode the DC coefficient difference */
	s = GetCategory( dctbl );		//get dc category number, s

	if (s) {
		r = GetBits(s);					//get offset in this dc category
		s = ValueFromCategory(s, r);	//get dc difference value
    }
	
    /* Convert DC difference to actual value, update last_dc_val */
    s += *pLastDC;
    *pLastDC = s;

    /* Output the DC coefficient (assumes jpeg_natural_order[0] = 0) */
    coef[0] = s;	
    
	/* Section F.2.2.2: decode the AC coefficients */
	/* Since zeroes are skipped, output area must be cleared beforehand */
	for (k = 1; k < 64; k++) 
	{
		s = GetCategory( actbl );	//s: (run, category)
		r = s >> 4;			//	r: run length for ac zero, 0 <= r < 16
		s &= 15;			//	s: category for this non-zero ac
		
		if( s ) 
		{
			k += r;					//	k: position for next non-zero ac
			r = GetBits(s);			//	r: offset in this ac category
			s = ValueFromCategory(s, r);	//	s: ac value

			coef[ jpeg_natural_order[ k ] ] = s;
		} 
		else // s = 0, means ac value is 0 ? Only if r = 15.  
		{
			if (r != 15)	//means all the left ac are zero
				break;
			k += 15;
		}
	}		
}



////////////////////////////////////////////////////////////////////////////////
//	function Purpose:	decompress one 16*16 pixels
//	source is m_pData;
//	This function will push m_pData ahead for next tile

bool DecompressOneTile(int * pDCT)
{
	// Process restart marker if needed; may have to suspend 
	if (restart_interval) 
	{
		if (restarts_to_go == 0)
		{
			m_nGetBits  = 0;
			read_restart_marker();
			gnJPEGDecoderdcY = gnJPEGDecoderdcCb = gnJPEGDecoderdcCr = 0;
			restarts_to_go = restart_interval;
		}
	}

	//	Do Y/Cb/Cr components, 
	//	if m_nBlocksInMcu==6,  Y: 4 blocks; Cb: 1 block; Cr: 1 block
	//	if m_nBlocksInMcu==3,  Y: 1 block; Cb: 1 block; Cr: 1 block
	for( int i=0; i<m_nBlocksInMcu; i++ )
	{
		HuffmanDecode( pDCT + i * 64, i );	
	}

	// Account for restart interval (no-op if not using restarts) 
	restarts_to_go--;

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//	Prepare for all the tables needed, 
//	eg. quantization tables, huff tables, color convert tables
//	1 <= nQuality <= 100, is used for quantization scaling
//	Computing once, and reuse them again and again !!!!!!!

void RTGPUInitDecoder( void )
{
	m_nGetBits = 0;
	m_nGetBuff = 0;

	gnJPEGDecoderdcY = gnJPEGDecoderdcCb = gnJPEGDecoderdcCr = 0;

	//	prepare range limiting table to limit idct outputs
	SetRangeTable( );

	//	prepare color convert table, from bgr to ycbcr
	RTGPUJPEGDecoderInitColorTable( );

	//	prepare two quant tables, one for Y, and another for CbCr
	RTGPUJPEGDecoderInitQuantTable( );

	//	prepare four huffman tables: 
	RTGPUJPEGDecoderInitHuffmanTable( );
}

////////////////////////////////////////////////////////////////////////////////

extern "C" bool _RTGPUReadJPEGHeader(	
	unsigned char *pInBuf,	//in, source data, in jpg format
	int cbInBuf,			//in, count bytes for in buffer
	int& nWidth,			//out, image width in pixels
	int& nHeight,			//out, image height
	int& nHeadSize			//out, header size in bytes
	)
{
	// Step 1:
	if( read_markers( pInBuf, cbInBuf, nWidth, nHeight, nHeadSize )==-1 )
	{
		RTGPUTrace("Cannot read the file header");
		return false;
	}
	if(( gnJPEGDecoderWidth <= 0 )||( gnJPEGDecoderHeight <= 0 ))
		return false;
	m_nDataBytesLeft = cbInBuf - nHeadSize;

	RTGPUInitDecoder();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//	DecompressImage(), the main function in this class !!
//	IMPORTANT: You should call ReadJPEGHeader() to get image width and height,
//				Then allocate (gnJPEGDecoderWidth * gnJPEGDecoderHeight * 3) bytes for pOutBuf

extern "C" bool _RTGPUJPEGDecompress(unsigned char *inBuf, int destSlot)
{
	int nW, nH;
	RTGPU_IMAGE	*pDI;
    cudaChannelFormatDesc desc;
	int		nYCbCrBlock;				// length of the YCbCr block (3 * 64 or 6 * 64)

	unsigned char	*pGPUYCBCR;			// GPU buffer for YCbCr data
	int		*pGPUIDCTC;					// GPU buffer for IDCT data col stage
	int		*pGPUIDCTR;					// GPU buffer for IDCT data row stage
	int		nYCbCrSize;					// total size of YCbCr data

	int		*pIDCT;						// used for the result of the Huffman decoding

	int xTile, yTile, cxTile, cyTile;

	RTGPU_SLOTPTR(destSlot, pDI);
	nW = pDI->width;
	nH = pDI->height;

	RTGPUTrace("RTGPUJPEGDecompress");

	_RTGPUSetupSlot(pDI, gnJPEGDecoderWidth, gnJPEGDecoderHeight, 3);
		
	//	horizontal and vertical count of tile, macroblocks, 
	//	MCU(Minimum Coded Unit), 
	//		case 1: maybe is 16*16 pixels, 6 blocks
	//		case 2: may be 8*8 pixels, only 3 blocks
	cxTile = (gnJPEGDecoderWidth  + m_nMcuSize - 1) / m_nMcuSize;	
	cyTile = (gnJPEGDecoderHeight + m_nMcuSize - 1) / m_nMcuSize;

	nYCbCrBlock = DCTSIZE * DCTSIZE * m_nBlocksInMcu;
	nYCbCrSize = cxTile * cyTile * nYCbCrBlock;
	pIDCT = (int *)malloc(nYCbCrSize * sizeof(int));
	RTGPUSafeCall(cudaMalloc(&(pGPUIDCTC), nYCbCrSize * sizeof(int)));
	RTGPUSafeCall(cudaMalloc(&(pGPUIDCTR), nYCbCrSize * sizeof(int)));
	RTGPUSafeCall(cudaMalloc(&(pGPUYCBCR), nYCbCrSize));

	//	source ptr
	m_pData = inBuf;

	//	Decompress all the tiles, or macroblocks, or MCUs
	for( yTile = 0; yTile < cyTile; yTile++ )
	{
		for( xTile = 0; xTile < cxTile; xTile++ )
		{
			if( ! DecompressOneTile( pIDCT + (yTile * cxTile + xTile) * nYCbCrBlock))
				return false;
		}
	}

	RTGPUSafeCall(cudaMemcpy(pGPUIDCTC, pIDCT, nYCbCrSize * sizeof(int), cudaMemcpyHostToDevice));

	desc = cudaCreateChannelDesc<int>();
	RTGPUSafeCall(cudaBindTexture(NULL, gpTexIDCT, pGPUIDCTC, desc, nYCbCrSize * sizeof(int)));

	dim3	threadsDCT(DCTSIZE, DCTBLOCKS);		// .x = 8, .y = 6

	kernelInverseDctCol<<<cxTile * cyTile, threadsDCT>>>(pGPUIDCTR);

	RTGPUSafeCall(cudaBindTexture(NULL, gpTexIDCT, pGPUIDCTR, desc, nYCbCrSize * sizeof(int)));
	
	kernelInverseDctRow<<<cxTile * cyTile, threadsDCT>>>(pGPUYCBCR);

	RTGPUSafeCall(cudaUnbindTexture(gpTexIDCT));

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaBindTexture(NULL, gpTexYCBCR, pGPUYCBCR, desc, nYCbCrSize));

	dim3	threadsConv(m_nMcuSize, m_nMcuSize);

	kernelYCbCrToRGB<<<cyTile, threadsConv>>>(pDI->image, cxTile, nW, nH); // Convert to RGB

	RTGPUSafeCall(cudaUnbindTexture(gpTexYCBCR));

	RTGPUSafeCall(cudaFree(pGPUYCBCR));
	RTGPUSafeCall(cudaFree(pGPUIDCTR));
	RTGPUSafeCall(cudaFree(pGPUIDCTC));
	free(pIDCT);
	return true;
}


////////////////////////////////////////////////////////////////////////////////
// end //
