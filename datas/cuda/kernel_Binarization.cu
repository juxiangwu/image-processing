
__global__ void Binarize(unsigned int *dst, int imageW, int imageH, int threshold)
{	
	
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    
	if(ix < imageW && iy < imageH){

		int pix = dst[imageW * iy + ix] & 0xff;
		if(pix < threshold) dst[imageW * iy + ix] =	make_color(0.f, 0.f, 0.f, 0.f);   //object pixel's value should be greater
		else dst[imageW * iy + ix] = make_color(1.f, 1.f, 1.f , 1.f);

	}	
}

extern "C" float binarizationWrapper (unsigned int *dst, int imageW, int imageH, int threshold, float brightness, float contrast, int adjust)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int timer;
	float runtime;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	if(adjust)
		Grayscale<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
	else
		Grayscale2<<<grid, threads>>>(dst, imageW, imageH);

	cudaThreadSynchronize();

	Binarize<<<grid, threads>>>(dst, imageW, imageH, threshold);
	
	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}