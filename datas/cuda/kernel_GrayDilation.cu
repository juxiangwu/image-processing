__global__ void GrayDilation (unsigned int *dst, int imageW, int imageH,  int mask_w, int mask_h)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){
		
		float4 fresult = tex2D(texUCHAR, ix, iy);
		unsigned int max = make_color(fresult.x, fresult.y, fresult.z , 1.f);
		unsigned int new_max = 0;

		for (int m = ix - mask_w+1 ; m < ix + mask_w-1; m++){			
			for (int n = iy - mask_h+1 ; n < iy + mask_h-1; n++){				
				fresult = tex2D(texUCHAR, m, n);
				new_max = make_color(fresult.x, fresult.y, fresult.z , 1.f);
				if (max < new_max) max = new_max;					
			}
		}
		  

		dst[imageW * iy + ix] = max;
	
	}
}

extern "C" float grayDilationWrapper (unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast,  int mask_w, int mask_h, int adjust)
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
	
	for(int i=0; i<iteration; i++)
	{

	cudaMemcpyToArray( d_tempArray, 0, 0, dst, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(texUCHAR, d_tempArray);

	GrayDilation<<<grid, threads>>>(dst, imageW, imageH,  mask_w, mask_h);

	}
	
	cudaUnbindTexture(texUCHAR);
	cudaThreadSynchronize();
	cutStopTimer(timer);

	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}