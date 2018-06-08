extern "C"{

    #define RGB2GRAY_CONST_ARR_SIZE 3
    #define STRONG_EDGE 255
    #define NON_EDGE 0.0
    #define KERNEL_SIZE 7
    //*****************************************************************************************
    // CUDA Gaussian Filter Implementation
    //*****************************************************************************************

    ///
    /// \brief Apply gaussian filter. This is the CUDA kernel for applying a gaussian blur to an image.
    ///
    __global__ void cu_apply_gaussian_filter(float3 *in_pixels, float3 *out_pixels, int rows, int cols, double *in_kernel)
    {
        //copy kernel array from global memory to a shared array
        __shared__ double kernel[KERNEL_SIZE][KERNEL_SIZE];
        for (int i = 0; i < KERNEL_SIZE; ++i) {
            for (int j = 0; j < KERNEL_SIZE; ++j) {
                kernel[i][j] = in_kernel[i * KERNEL_SIZE + j];
            }
        }

        __syncthreads();

        //determine id of thread which corresponds to an individual pixel
        int pixNum = blockIdx.x * blockDim.x + threadIdx.x;
        if (pixNum >= 0 && pixNum < rows * cols) {

            double kernelSum;
            double redPixelVal;
            double greenPixelVal;
            double bluePixelVal;

            //Apply Kernel to each pixel of image
            for (int i = 0; i < KERNEL_SIZE; ++i) {
                for (int j = 0; j < KERNEL_SIZE; ++j) {    

                    //check edge cases, if within bounds, apply filter
                    if (((pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)) >= 0)
                        && ((pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)) <= rows*cols-1)
                        && (((pixNum % cols) + j - ((KERNEL_SIZE-1)/2)) >= 0)
                        && (((pixNum % cols) + j - ((KERNEL_SIZE-1)/2)) <= (cols-1))) {

                        redPixelVal += kernel[i][j] * in_pixels[pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)].x;
                        greenPixelVal += kernel[i][j] * in_pixels[pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)].y;
                        bluePixelVal += kernel[i][j] * in_pixels[pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)].z;
                        kernelSum += kernel[i][j];
                    }
                }
            }

            //update output image
            out_pixels[pixNum].x = redPixelVal / kernelSum;
            out_pixels[pixNum].y = greenPixelVal / kernelSum;
            out_pixels[pixNum].z = bluePixelVal / kernelSum;
        }
    }
    
    //*****************************************************************************************
    // CUDA Intensity Gradient Implementation
    //*****************************************************************************************

    ///
    /// \brief Compute gradient (first order derivative x and y). This is the CUDA kernel for taking the derivative of color contrasts in adjacent images.
    ///
    __global__
    void cu_compute_intensity_gradient(float3 *in_pixels, float *deltaX_channel, float *deltaY_channel, int parser_length, int offset)
    {
        // compute delta X ***************************
        // deltaX = f(x+1) - f(x-1)

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        /* condition here skips first and last row */
        if ((idx > offset) && (idx < (parser_length * offset) - offset))
        {
            float deltaXred = 0;
            float deltaYred = 0;
            float deltaXgreen = 0;
            float deltaYgreen = 0;
            float deltaXblue = 0;
            float deltaYblue = 0;

            /* first column */
            if((idx % offset) == 0)
            {
                // gradient at the first pixel of each line
                // note: at the edge pix[idx-1] does NOT exist
                deltaXred = (float)(in_pixels[idx+1].x - in_pixels[idx].x);
                deltaXgreen = (float)(in_pixels[idx+1].y - in_pixels[idx].y);
                deltaXblue = (float)(in_pixels[idx+1].z - in_pixels[idx].z);
                // gradient at the first pixel of each line
                // note: at the edge pix[idx-1] does NOT exist
                deltaYred = (float)(in_pixels[idx+offset].x - in_pixels[idx].x);
                deltaYgreen = (float)(in_pixels[idx+offset].y - in_pixels[idx].y);
                deltaYblue = (float)(in_pixels[idx+offset].z - in_pixels[idx].z);
            }
            /* last column */
            else if((idx % offset) == (offset - 1))
            {
                deltaXred = (float)(in_pixels[idx].x - in_pixels[idx-1].x);
                deltaXgreen = (float)(in_pixels[idx].y - in_pixels[idx-1].y);
                deltaXblue = (float)(in_pixels[idx].z - in_pixels[idx-1].z);
                deltaYred = (float)(in_pixels[idx].x - in_pixels[idx-offset].x);
                deltaYgreen = (float)(in_pixels[idx].y - in_pixels[idx-offset].y);
                deltaYblue = (float)(in_pixels[idx].z - in_pixels[idx-offset].z);
            }
            /* gradients where NOT edge */
            else
            {
                deltaXred = (float)(in_pixels[idx+1].x - in_pixels[idx-1].x);
                deltaXgreen = (float)(in_pixels[idx+1].y - in_pixels[idx-1].y);
                deltaXblue = (float)(in_pixels[idx+1].z - in_pixels[idx-1].z);
                deltaYred = (float)(in_pixels[idx+offset].x - in_pixels[idx-offset].x);
                deltaYgreen = (float)(in_pixels[idx+offset].y - in_pixels[idx-offset].y);
                deltaYblue = (float)(in_pixels[idx+offset].z - in_pixels[idx-offset].z);
            }
            deltaX_channel[idx] = (float)(0.2989 * deltaXred + 0.5870 * deltaXgreen + 0.1140 * deltaXblue);
            deltaY_channel[idx] = (float)(0.2989 * deltaYred + 0.5870 * deltaYgreen + 0.1140 * deltaYblue); 
        }
}

    //*****************************************************************************************
    // CUDA Gradient Magnitude Implementation
    //*****************************************************************************************

    ///
    /// \brief Compute magnitude of gradient(deltaX & deltaY) per pixel.
    ///
    __global__
    void cu_magnitude(float *deltaX, float *deltaY, float *out_pixel, int parser_length, int offset)
    {
        //computation
        //Assigned a thread to each pixel
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= 0 && idx < parser_length * offset) {
                out_pixel[idx] =  (float)(sqrt((double)deltaX[idx]*deltaX[idx] + 
                                (double)deltaY[idx]*deltaY[idx]) + 0.5);
            }
    }
    
    
    //*****************************************************************************************
    // CUDA Non Maximal Suppression Implementation
    //*****************************************************************************************

    ///
    /// \brief Non Maximal Suppression
    /// If the centre pixel is not greater than neighboured pixels in the direction,
    /// then the center pixel is set to zero.
    /// This process results in one pixel wide ridges.
    ///
    __global__ void cu_suppress_non_max(float *mag, float *deltaX, float *deltaY, float *nms, int parser_length, int offset)
    {

        const float SUPPRESSED = 0;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= 0 && idx < parser_length * offset)
        {
            float alpha;
            float mag1, mag2;
            // put zero all boundaries of image
            // TOP edge line of the image
            if((idx >= 0) && (idx <offset))
                nms[idx] = 0;

            // BOTTOM edge line of image
            else if((idx >= (parser_length-1)*offset) && (idx < (offset * parser_length)))
                nms[idx] = 0;

            // LEFT & RIGHT edge line
            else if(((idx % offset)==0) || ((idx % offset)==(offset - 1)))
            {
                nms[idx] = 0;
            }

            else // not the boundaries
            {
                // if magnitude = 0, no edge
                if(mag[idx] == 0)
                    nms[idx] = SUPPRESSED;
                    else{
                        if(deltaX[idx] >= 0)
                        {
                            if(deltaY[idx] >= 0)  // dx >= 0, dy >= 0
                            {
                                if((deltaX[idx] - deltaY[idx]) >= 0)       // direction 1 (SEE, South-East-East)
                                {
                                    alpha = (float)deltaY[idx] / deltaX[idx];
                                    mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                                    mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                                }
                                else                                // direction 2 (SSE)
                                {
                                    alpha = (float)deltaX[idx] / deltaY[idx];
                                    mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                                    mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                                }
                            }
                            else  // dx >= 0, dy < 0
                            {
                                if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 8 (NEE)
                                {
                                    alpha = (float)-deltaY[idx] / deltaX[idx];
                                    mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                                    mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                                }
                                else                                // direction 7 (NNE)
                                {
                                    alpha = (float)deltaX[idx] / -deltaY[idx];
                                    mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                                    mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                                }
                            }
                        }

                        else
                        {
                            if(deltaY[idx] >= 0) // dx < 0, dy >= 0
                            {
                                if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 3 (SSW)
                                {
                                    alpha = (float)-deltaX[idx] / deltaY[idx];
                                    mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                                    mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                                }
                                else                                // direction 4 (SWW)
                                {
                                    alpha = (float)deltaY[idx] / -deltaX[idx];
                                    mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                                    mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                                }
                            }

                            else // dx < 0, dy < 0
                            {
                                 if((-deltaX[idx] + deltaY[idx]) >= 0)   // direction 5 (NWW)
                                 {
                                     alpha = (float)deltaY[idx] / deltaX[idx];
                                     mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                                     mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                                 }
                                 else                                // direction 6 (NNW)
                                 {
                                     alpha = (float)deltaX[idx] / deltaY[idx];
                                     mag1 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                                     mag2 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                                 }
                            }
                        }

                        // non-maximal suppression
                        // compare mag1, mag2 and mag[t]
                        // if mag[t] is smaller than one of the neighbours then suppress it
                        if((mag[idx] < mag1) || (mag[idx] < mag2))
                             nms[idx] = SUPPRESSED;
                        else
                        {
                             nms[idx] = mag[idx];
                        }

                } // END OF ELSE (mag != 0)
            } // END OF FOR(j)
        } // END OF FOR(i)
    }
    
    //*****************************************************************************************
    // CUDA Hysteresis Implementation
    //*****************************************************************************************

    ///
    /// \brief This is a helper function that runs on the GPU.
    ///
    /// It checks if the eight immediate neighbors of a pixel at a given index are above
    /// a low threshold, and if they are, sets them to strong edges. This effectively
    /// connects the edges.
    ///
    __device__
    void trace_immed_neighbors(float *out_pixels, float *in_pixels, 
                                int idx, float t_low, int img_width)
    {
        /* directions representing indices of neighbors */
        unsigned n, s, e, w;
        unsigned nw, ne, sw, se;

        /* get indices */
        n = idx - img_width;
        nw = n - 1;
        ne = n + 1;
        s = idx + img_width;
        sw = s - 1;
        se = s + 1;
        w = idx - 1;
        e = idx + 1;

        if (in_pixels[nw] >= t_low &&in_pixels[nw]!=255.0 ) {
            out_pixels[nw] = STRONG_EDGE;
        }
        if (in_pixels[n] >= t_low&&in_pixels[n]!=255.0) {
            out_pixels[n] = STRONG_EDGE;
        }
        if (in_pixels[ne] >= t_low&&in_pixels[ne]!=255.0) {
            out_pixels[ne] = STRONG_EDGE;
        }
        if (in_pixels[w] >= t_low&&in_pixels[w]!=255.0) {
            out_pixels[w] = STRONG_EDGE;
        }
        if (in_pixels[e] >= t_low&&in_pixels[e]!=255.0) {
            out_pixels[e] = STRONG_EDGE;
        }
        if (in_pixels[sw] >= t_low&&in_pixels[sw]!=255.0) {
            out_pixels[sw] = STRONG_EDGE;
        }
        if (in_pixels[s] >= t_low&&in_pixels[s]!=255.0) {
            out_pixels[s] = STRONG_EDGE;
        }
        if (in_pixels[se] >= t_low&&in_pixels[se]!=255.0) {
            out_pixels[se] = STRONG_EDGE;
        }
    }
    
    ///
    /// \brief CUDA implementation of Canny hysteresis high thresholding.
    ///
    /// This kernel is the first pass in the parallel hysteresis step.
    /// It launches a thread for every pixel and checks if the value of that pixel
    /// is above a high threshold. If it is, the thread marks it as a strong edge (set to 1)
    /// in a pixel map and sets the value to the channel max. If it is not, the thread sets
    /// the pixel map at the index to 0 and zeros the output buffer space at that index.
    ///
    /// The output of this step is a mask of strong edges and an output buffer with white values
    /// at the mask indices which are set.
    ///
    __global__
    void cu_hysteresis_high(float *out_pixels, float *in_pixels, float *strong_edge_mask, 
                            float t_high, int img_height, int img_width)
    {
        //printf("t_high=%f\n",t_high);
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < (img_height * img_width)) {
            /* apply high threshold */
          
           //printf("pixel=%f\n",in_pixels[idx]);
           
            if (in_pixels[idx] > t_high) {
                strong_edge_mask[idx] = 1.0;
                out_pixels[idx] = STRONG_EDGE;
            } else {
                strong_edge_mask[idx] = 0.0;
                out_pixels[idx] = NON_EDGE;
            }
        }
    }

    ///
    /// \brief CUDA implementation of Canny hysteresis low thresholding.
    ///
    /// This kernel is the second pass in the parallel hysteresis step. 
    /// It launches a thread for every pixel, but skips the first and last rows and columns.
    /// For surviving threads, the pixel at the thread ID index is checked to see if it was 
    /// previously marked as a strong edge in the first pass. If it was, the thread checks 
    /// their eight immediate neighbors and connects them (marks them as strong edges)
    /// if the neighbor is above the low threshold.
    ///
    /// The output of this step is an output buffer with both "strong" and "connected" edges
    /// set to whtie values. This is the final edge detected image.
    ///
    __global__
    void cu_hysteresis_low(float *out_pixels, float *in_pixels, float *strong_edge_mask,
                            float t_low, int img_height, int img_width)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ((idx > img_width)                               /* skip first row */
            && (idx < (img_height * img_width) - img_width) /* skip last row */
            && ((idx % img_width) < (img_width - 1))        /* skip last column */
            && ((idx % img_width) > (0)) )                  /* skip first column */
        {
            if (1.0 == strong_edge_mask[idx]) { /* if this pixel was previously found to be a strong edge */
                trace_immed_neighbors(out_pixels, in_pixels, idx, t_low, img_width);
            }
        }
    }
    
   __global__ void hysteresis_kernel(float* out_pixels,float * in_pixels,float t_low,float t_high,int img_height,int img_width){
   
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if ((idx > img_width) &&
         (idx < (img_height * img_width) - img_width) &&
         ((idx % img_width) < (img_width - 1)) &&
         ((idx % img_width) > (0)) ){
         
             float pixel = in_pixels[idx];
             if (pixel != 255.0){
               if (pixel > t_high){
                  out_pixels[idx] = (float)255.0;
                  trace_immed_neighbors(out_pixels,in_pixels,idx,t_low,img_width);
               }else{
                out_pixels[idx] = 0.0;
               }
             }
         }
   
 }
    


}