__device__ float highpass (float p00, float p01, float p02, 
					  float p10, float p11, float p12, 
					  float p20, float p21, float p22) 
{
	float result = (9*p11 -p00 -p01 -p02 -p10 -p12 -p20 -p21 -p22);
	//float result = (4*p00+(-4)*p22);	
	//float result = -p00 - p01 + p02 - p10 - p11 + p12 + p20 + p21 + p22;
	if(result< 0.f) return 0.f; else if(result>1.0f) return 1.0f;
	return result;
}

__global__ void HighPassFilter(unsigned int *dst, int imageW, int imageH)
{	
	
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH) {
		float4 rgba;

		float pix00 = (tex2D( texUCHAR, (float) ix-1, (float) iy-1 ).x);
		float pix01 = (tex2D( texUCHAR, (float) ix+0, (float) iy-1 ).x);
		float pix02 = (tex2D( texUCHAR, (float) ix+1, (float) iy-1 ).x);
		float pix10 = (tex2D( texUCHAR, (float) ix-1, (float) iy+0 ).x);
		float pix11 = (tex2D( texUCHAR, (float) ix+0, (float) iy+0 ).x);
		float pix12 = (tex2D( texUCHAR, (float) ix+1, (float) iy+0 ).x);
		float pix20 = (tex2D( texUCHAR, (float) ix-1, (float) iy+1 ).x);
		float pix21 = (tex2D( texUCHAR, (float) ix+0, (float) iy+1 ).x);
		float pix22 = (tex2D( texUCHAR, (float) ix+1, (float) iy+1 ).x);
			
		rgba.x = highpass( pix00, pix01, pix02, 
								pix10, pix11, pix12,
								pix20, pix21, pix22 );


		pix00 = (tex2D( texUCHAR, (float) ix-1, (float) iy-1 ).y);
		pix01 = (tex2D( texUCHAR, (float) ix+0, (float) iy-1 ).y);
		pix02 = (tex2D( texUCHAR, (float) ix+1, (float) iy-1 ).y);
		pix10 = (tex2D( texUCHAR, (float) ix-1, (float) iy+0 ).y);
		pix11 = (tex2D( texUCHAR, (float) ix+0, (float) iy+0 ).y);
		pix12 = (tex2D( texUCHAR, (float) ix+1, (float) iy+0 ).y);
		pix20 = (tex2D( texUCHAR, (float) ix-1, (float) iy+1 ).y);
		pix21 = (tex2D( texUCHAR, (float) ix+0, (float) iy+1 ).y);
		pix22 = (tex2D( texUCHAR, (float) ix+1, (float) iy+1 ).y);
			
		rgba.y = highpass( pix00, pix01, pix02, 
								pix10, pix11, pix12,
								pix20, pix21, pix22 );

		pix00 = (tex2D( texUCHAR, (float) ix-1, (float) iy-1 ).z);
		pix01 = (tex2D( texUCHAR, (float) ix+0, (float) iy-1 ).z);
		pix02 = (tex2D( texUCHAR, (float) ix+1, (float) iy-1 ).z);
		pix10 = (tex2D( texUCHAR, (float) ix-1, (float) iy+0 ).z);
		pix11 = (tex2D( texUCHAR, (float) ix+0, (float) iy+0 ).z);
		pix12 = (tex2D( texUCHAR, (float) ix+1, (float) iy+0 ).z);
		pix20 = (tex2D( texUCHAR, (float) ix-1, (float) iy+1 ).z);
		pix21 = (tex2D( texUCHAR, (float) ix+0, (float) iy+1 ).z);
		pix22 = (tex2D( texUCHAR, (float) ix+1, (float) iy+1 ).z);
			
		rgba.z = highpass( pix00, pix01, pix02, 
								pix10, pix11, pix12,
								pix20, pix21, pix22 );
		
		dst[imageW * iy + ix] =	make_color(rgba.x, rgba.y, rgba.z, 1.f);
	
	}
}


extern "C" float highPassFilterWrapper (unsigned int *dst, int imageW, int imageH, int iteration, float brightness, float contrast, int adjust)
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

	for(int i=0; i<iteration; i++){
		cudaMemcpyToArray( d_tempArray, 0, 0, dst, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(texUCHAR, d_tempArray);

		HighPassFilter<<<grid, threads>>>(dst, imageW, imageH);
	}

	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}