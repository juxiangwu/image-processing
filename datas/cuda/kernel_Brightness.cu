__global__ void Brightness (unsigned int *dst, int imageW, int imageH, float brightness)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		float4 fresult = tex2D(texImage, ix, iy);
		float red = fresult.x;
		float green = fresult.y;
		float blue = fresult.z;

		red = red * (1.f - brightness) + brightness;
		green = green * (1.f - brightness) + brightness;
		blue = blue * (1.f - brightness) + brightness;

		dst[imageW * iy + ix] =  make_color(red, green, blue, 1.f);
	}
}


// if gamma is 0..1 , the dark intensities are stretched up
// if gamma is 1..5 , the high intensities are stretched down

extern "C" float brightnessWrapper (unsigned int *dst, int imageW, int imageH, float brightness)
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int timer;
	float runtime;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	
	Brightness<<<grid, threads>>>(dst, imageW, imageH, brightness);

	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}