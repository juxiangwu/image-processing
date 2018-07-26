__device__ float Sobel (float p00, float p01, float p02, 
					  float p10, float p11, float p12, 
					  float p20, float p21, float p22) 
{
	float Gx = p02 + 2*p12 + p22 - p00 - 2*p10 - p20;
    float Gy = p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
    float G = (abs(Gx)+abs(Gy));
    if ( G < 0 ) return 0.f; else if ( G > 1.f ) return 1.f;
    return G;

}

__global__ void SobelFilter(unsigned int *dst, int imageW, int imageH)
{	
	
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH) {

		float pix00 = (tex2D( texImage, (float) ix-1, (float) iy-1 ).x);
		float pix01 = (tex2D( texImage, (float) ix+0, (float) iy-1 ).x);
		float pix02 = (tex2D( texImage, (float) ix+1, (float) iy-1 ).x);
		float pix10 = (tex2D( texImage, (float) ix-1, (float) iy+0 ).x);
		float pix11 = (tex2D( texImage, (float) ix+0, (float) iy+0 ).x);
		float pix12 = (tex2D( texImage, (float) ix+1, (float) iy+0 ).x);
		float pix20 = (tex2D( texImage, (float) ix-1, (float) iy+1 ).x);
		float pix21 = (tex2D( texImage, (float) ix+0, (float) iy+1 ).x);
		float pix22 = (tex2D( texImage, (float) ix+1, (float) iy+1 ).x);
			
		float sobel = Sobel(	pix00, pix01, pix02, 
								pix10, pix11, pix12,
								pix20, pix21, pix22 );

		dst[imageW * iy + ix] =	make_color(sobel, sobel, sobel, 1.f);
	}
	
}


extern "C" float sobelFilterWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int timer;
	float runtime;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	if(adjust){
		Grayscale<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
	}else{
		Grayscale2<<<grid, threads>>>(dst, imageW, imageH);
	}

	SobelFilter<<<grid, threads>>>(dst, imageW, imageH);

	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}