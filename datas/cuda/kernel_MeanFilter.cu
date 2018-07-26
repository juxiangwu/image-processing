__global__ void MeanFilter ( unsigned int *dst, int imageW, int imageH, int radius)
{
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	float4 fresult = {0,0,0,0};
	float4 sum = {0,0,0,0};
	float count = 1.f;

	if(ix < imageW && iy < imageH) {

		for (int m = ix - radius ; m <= ix + radius; m++){			
			for (int n = iy - radius ; n <= iy + radius; n++){				
				fresult = tex2D(texUCHAR, m, n);
				sum.x += fresult.x;
				sum.y += fresult.y;
				sum.z += fresult.z;
				count+=1.f;
			}
		}

		sum.x /= count;
		sum.y /= count;
		sum.z /= count;

		dst[imageW * iy + ix] = make_color(sum.x, sum.y, sum.z, 1.f);

	}

}

extern "C" float meanFilterWrapper (unsigned int *dst, int imageW, int imageH, int radius, int iteration, float brightness, float contrast, int adjust) 
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	unsigned int timer;
	float runtime;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	Copy<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);

	cudaThreadSynchronize();

	for(int i=0; i<iteration; i++){
		cudaMemcpyToArray( d_tempArray, 0, 0, dst, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(texUCHAR, d_tempArray);

		MeanFilter<<<grid, threads>>>(dst, imageW, imageH, 3);
	}

	cudaThreadSynchronize();
	cutStopTimer(timer);
	
	runtime = cutGetTimerValue(timer)/1000;
	cutDeleteTimer(timer);

	return runtime;
}