__global__ void Invert2 (unsigned int *dst, int imageW, int imageH)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		float4 fresult = tex2D(texImage, ix, iy);
		dst[imageW * iy + ix] =  make_color(1.f-fresult.x, 1.f-fresult.y, 1.f-fresult.z, 1.f);
	}
}

__global__ void Invert (unsigned int *dst, int imageW, int imageH, float brightness, float contrast)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		float4 fresult = tex2D(texImage, ix, iy);
		//adjust brightness
		float red = fresult.x * (1.f - brightness) + brightness;
		float green = fresult.y * (1.f - brightness) + brightness;
		float blue = fresult.z * (1.f - brightness) + brightness;

		//adjust contrast
		red = pow(red, contrast);
		green = pow(green, contrast);
		blue = pow(blue, contrast);
		
		red = 1.f - red;
		green = 1.f - green;
		blue = 1.f - blue;

		dst[imageW * iy + ix] =  make_color(red, green, blue, 1.f);
	}
}

extern "C" float invertWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int timer;
	float runtime;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	if(adjust)
		Invert<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
	else
		Invert2<<<grid, threads>>>(dst, imageW, imageH);
	
	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}