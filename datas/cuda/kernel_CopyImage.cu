
__global__ void Copy2 ( unsigned int *dst, int imageW, int imageH) 
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if(ix < imageW && iy < imageH){
	    
		float4 fnew = tex2D(texImage, ix, iy);
		dst[imageW * iy + ix] =  make_color(
									fnew.x, 
									fnew.y, 
									fnew.z, 
									fnew.w
								 );        
    }
}

__global__ void Copy ( unsigned int *dst, int imageW, int imageH, float brightness, float contrast) 
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if(ix < imageW && iy < imageH){
	    
		float4 fresult = tex2D(texImage, ix, iy);
		float4 fnew = adjust_contrast(fresult, contrast);
		fnew = adjust_brightness(fnew, brightness);

		dst[imageW * iy + ix] =  make_color(
									fnew.x, 
									fnew.y, 
									fnew.z, 
									fnew.w
								 );        
    }
}

extern "C" float copyImageWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust) 
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int timer;
	float runtime;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	if(adjust)
		Copy<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
	else
		Copy2<<<grid, threads>>>(dst, imageW, imageH);

	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}