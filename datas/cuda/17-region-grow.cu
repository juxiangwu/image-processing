extern "C" {

    __global__ void fullgrow_kernel(double* d_image, double* d_region, double* d_conv, int h, int w)
    {
        int j = blockDim.x * blockIdx.x + threadIdx.x;
        int i = blockDim.y * blockIdx.y + threadIdx.y;
        int index = i*w + j;
        if ((0 < i) && (i < (h - 1)) && (0 < j) && (j < (w - 1))) {
            if (d_image[index] > 0.0 && d_image[index] < .8) {
                if (d_region[index + 1] == 1. || d_region[index - 1] == 1. || d_region[index + w] == 1. || d_region[index - w] == 1.) {
                    d_region[index] = 1.;
                    d_conv[index] = 1.;
                }
            }
        }
    }
    
    
    __global__ void findseeds_kernel(double* d_image, double* d_region, int h, int w) {
        int j = blockDim.x * blockIdx.x + threadIdx.x;
        int i = blockDim.y * blockIdx.y + threadIdx.y;
        int index = i*w + j;
        if ((0 < i) && (i < (h - 1)) && (0 < j) && (j < (w - 1))) {
            if (d_image[index] > 0.0 && d_image[index] < .8) {
                d_region[index] = 1;
            }
        }
    }

}