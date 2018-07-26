extern "C" {

// kernel function to blur a single color channel (R || G || B)
    __global__ void convolute(float *ch, float *res, int height, int width,int ksize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // width index
        int y = blockIdx.y * blockDim.y + threadIdx.y; // height index

        //int radius = 8;
        float PI = atanf(1) * 4;
        if ((x < width) && (y < height)) {
            float sum = 0;
            float val = 0;
            int idx = x * width + y; // current pixel index
           int radius = ksize / 2;
            for (int i = y - radius; i < y + radius + 1; i++) {
                for (int j = x - radius; j < x + radius + 1; j++) {
                  if ((y-radius) > 0 && (y+radius) < height && (x - radius) > 0 && (x+radius)<width){ 
                    int h = fminf(height - 1, fmaxf(0, i));
                    int w = fminf(width - 1, fmaxf(0, j));
                    int dsq = (j - x) * (j - x) + (i - y) * (i - y);
                    float wght = expf(-dsq / (2 * radius * radius)) / (PI * 2 * radius * radius);

                    val += ch[w * width + h] * wght;
                    sum += wght;
                    }
                }
            }
            res[idx] = round(val / sum);
        }
    }
}