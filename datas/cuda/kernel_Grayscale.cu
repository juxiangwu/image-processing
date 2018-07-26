__global__ void Grayscale2 ( unsigned int *dst, int imageW, int imageH) 
{
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		float4 fnew = tex2D(texImage, ix, iy);
		float gray = (fnew.x + fnew.y + fnew.z)/3;
        dst[imageW * iy + ix] = make_color(gray, gray, gray, 1.0f);
	}

}

__global__ void Grayscale ( unsigned int *dst, int imageW, int imageH, float brightness, float contrast) 
{
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		float4 fresult = tex2D(texImage, ix, iy);
		float4 fnew = adjust_contrast(fresult, contrast);
		fnew = adjust_brightness(fnew, brightness);

		float gray = (fnew.x + fnew.y + fnew.z)/3;
        dst[imageW * iy + ix] = make_color(gray, gray, gray, 1.0f);
	}

}

extern "C" float grayImageWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust) 
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
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}