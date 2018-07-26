__global__ void BinaryErosion (unsigned int *dst, int imageW, int imageH,  int mask_w, int mask_h)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){
		
		int match = 0;
		for (int m = ix - mask_w ; m < ix + mask_w && !match; m++){
			for (int n = iy - mask_h ; n < iy + mask_h && !match; n++){
				float4 fresult = tex2D(texUCHAR, m, n);
				if (fresult.x == 1.f && fresult.y == 1.f && fresult.z == 1.f )
					match = 1;
			}
		} 

		if(!match)
		dst[imageW * iy + ix] = make_color(0.f, 0.f, 0.f , 1.f);
		else
		dst[imageW * iy + ix] = make_color(1.f, 1.f, 1.f , 1.f);
	
	}
}


extern "C" float binaryErosionWrapper (unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast,  int mask_w, int mask_h, int adjust)
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

	Binarize<<<grid, threads>>>(dst, imageW, imageH, threshold);
	
	for(int i=0; i<iteration; i++)
	{

		cudaMemcpyToArray( d_tempArray, 0, 0, dst, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(texUCHAR, d_tempArray);
		
		BinaryErosion<<<grid, threads>>>(dst, imageW, imageH,  mask_w, mask_h);

	}	

	cudaUnbindTexture(texUCHAR);

	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}