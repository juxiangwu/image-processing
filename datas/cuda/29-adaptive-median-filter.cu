extern "C" {

    #define MAX_AREA_SIZE 7
    #define MEDIAN_BUFFER_SIZE (MAX_AREA_SIZE * MAX_AREA_SIZE + 1)


    __device__ void quickSort(unsigned char *arr, int left, int right) {
        int i = left, j = right;
        int tmp;
        int pivot = arr[(left + right) / 2];

        /* partition */
        while (i <= j) {
            while (arr[i] < pivot)
                i++;
            while (arr[j] > pivot)
                j--;
            if (i <= j) {
                tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
                i++;
                j--;
            }
        };

        /* recursion */
        if (left < j)
            quickSort(arr, left, j);
        if (i < right)
            quickSort(arr, i, right);
    }
    
    __global__ void adaptive_median_filter_kernel(unsigned char *imageData, unsigned char *filteredImageData, int width,int height,
        unsigned char *medianBuffer)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        //int width = gridDim.x;
        //int height = gridDim.y;

        bool processed = false;

        int pixelOffset = y * width + x;

        unsigned int pixel = imageData[pixelOffset];

       
        int n = 3;

        unsigned char *median = medianBuffer + ((y * width) + x) * MEDIAN_BUFFER_SIZE;

        //thrust::device_vector<unsigned char> median(MAX_AREA_SIZE * MAX_AREA_SIZE + 1, 255);

        //std::array<unsigned char, MAX_AREA_SIZE * MAX_AREA_SIZE + 1> median;

        while (!processed) {
           
            double zMin = 255;
           
            double zMax = 0;
           
            double zMed = 0;

            
            int sDelta = (n - 1) / 2;

            int processedPixelCount = 0;

            
            for (int sx = x - sDelta; sx <= x + sDelta; sx++) {
                for (int sy = y - sDelta; sy <= y + sDelta; sy++) {
                    if (sx < 0 || sy < 0 || sx >= width || sy >= height) {
                        continue;
                    }

                    unsigned int currentPixel = imageData[sy * width + sx];

                    if (currentPixel < zMin) {
                        zMin = currentPixel;
                    }

                    if (currentPixel > zMax) {
                        zMax = currentPixel;
                    }

                    median[processedPixelCount] = currentPixel;

                    processedPixelCount++;
                }
            }

            quickSort(median, 0, processedPixelCount);

            zMed = median[processedPixelCount / 2];

            double a1 = zMed - zMin;
            double a2 = zMed - zMax;

            if (a1 > 0 && a2 < 0) {
                double b1 = pixel - zMin;
                double b2 = pixel - zMax;

                if (b1 > 0 && b2 < 0) {
                    filteredImageData[pixelOffset] = pixel;
                }
                else {
                    filteredImageData[pixelOffset] = zMed;
                }

                processed = true;
            }
            else {
                n += 2;
                if (n > 7) {
                    filteredImageData[pixelOffset] = zMed;
                    processed = true;
                }
            }
        }
    }

}