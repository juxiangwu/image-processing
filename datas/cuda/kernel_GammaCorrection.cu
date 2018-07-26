__global__ void GammaCorrection (unsigned int *dst, int imageW, int imageH, float gamma)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		float4 fresult = tex2D(texImage, ix, iy);
		float red = pow(fresult.x, gamma);
		float green = pow(fresult.y, gamma);
		float blue = pow(fresult.z, gamma);

		dst[imageW * iy + ix] =  make_color(red, green, blue, 1.f);
	}
}


// if gamma is 0..1 , the dark intensities are stretched up
// if gamma is 1..5 , the high intensities are stretched down

extern "C" float gammaCorrectionWrapper (unsigned int *dst, int imageW, int imageH, float gamma)
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int timer;
	float runtime;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	GammaCorrection<<<grid, threads>>>(dst, imageW, imageH, gamma);

	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}