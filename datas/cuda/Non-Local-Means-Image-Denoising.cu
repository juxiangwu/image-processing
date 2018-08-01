#include<iostream>
#include<math.h>

using namespace std;
#define imgH 100
#define imgW 100
#define h 1
#define patchW 3

void printImg(float img[imgH][imgW])
{
    for(int i=0; i<imgH; i++)
    {
      for(int j=0; j<imgW; j++)
      {
        cout<<img[i][j]<<" ";
      }

      cout<<endl;
    }
  return;
}

__global__ void NLM(float* img, float* imgTemp, float* C)
{
  int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int j = by*blockDim.y + ty;
	int i = bx*blockDim.x + tx;

  for(int k=i; k<imgH - patchW + 1; k++)
    for(int l=0; l<imgW - patchW + 1; l++)
    {

      if(l != j)
      {
        float v = 0;

        for(int p=k; p<k+patchW; p++)
          for(int q=l; q<l+patchW; q++)
          {
            v += (img[(i+p-k)*imgW + j+q-l] - img[p*imgW + q]);
            v = v*v;
          }

          float w = exp(-v/(h*h));

          atomicAdd(imgTemp + i*imgW + j, w * img[k*imgW + l]);
          atomicAdd(C + i*imgW + j,  w);
          atomicAdd(imgTemp + k*imgW + l, w * img[i*imgW + j]);
          atomicAdd(C + k*imgW + l, w);
        }
      }
}


int main()
{
  float img[imgH][imgW] = {0}, C[imgH][imgW] = {0};
  float imgTemp[imgH][imgW] = {0};
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int i=0; i<imgH; i++)
    for(int j=0; j<imgW; j++)
    {
      img[i][j] = 1.2;
    }

    float* deviceImg;
    float* deviceImgTemp;
    float* deviceC;

    //Memory Allocation
    cudaMalloc((void**)&deviceImg, imgH * imgW * sizeof(float));
    cudaMalloc((void**)&deviceImgTemp, imgH * imgW * sizeof(float));
    cudaMalloc((void**)&deviceC, imgH * imgW * sizeof(float));
    cudaMemset(deviceImgTemp, 0, imgH * imgW * sizeof(float));
    cudaMemset(deviceC, 0, imgH * imgW * sizeof(float));

    //Memory Copy
    cudaMemcpy(deviceImg, img, imgH * imgW * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(1, 1, 1);
    dim3 grid(imgH - patchW + 1, imgW - patchW + 1, 1);

    cudaEventRecord(start);
    //Invoke Kernel
    NLM<<<grid, block>>>(deviceImg, deviceImgTemp, deviceC);
    cudaEventRecord(stop);

    //Memory
    cudaMemcpy(imgTemp, deviceImgTemp, imgH * imgW * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, deviceC, imgH * imgW * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    for(int i=0; i<imgH - patchW + 1; i++)
      for(int j=0; j<imgW - patchW + 1; j++)
      {
        img[i][j] = imgTemp[i][j]/C[i][j];
      }

  cout<<"Time elapsed for H = "<<imgH<<" and W = "<<imgW<<" is: "<<(float)milliseconds<<" milliseconds";

//  printImg(img);

  cudaFree(deviceImg);
  cudaFree(deviceImgTemp);
  cudaFree(deviceC);

  return 0;
}