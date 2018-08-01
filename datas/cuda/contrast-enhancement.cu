#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"

__device__ unsigned char GPU_clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

__device__ float GPU_Hue_2_RGB( float v1, float v2, float vH )
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

__global__ void GPU_yuv2rgb_kernel(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *y, unsigned char *u, unsigned char *v, int img_size)
{
    int rt, gt, bt;
    int Y, cb, cr;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < img_size)
    {
        Y  = (int)y[i];
        cb = (int)u[i] - 128;
        cr = (int)v[i] - 128;
        
        rt  = (int)(Y + 1.402 * cr);
        gt  = (int)(Y - 0.344 * cb - 0.714 * cr);
        bt  = (int)(Y + 1.772 * cb);

        r[i] = GPU_clip_rgb(rt);
        g[i] = GPU_clip_rgb(gt);
        b[i] = GPU_clip_rgb(bt);
    }
}

__global__ void GPU_rgb2yuv_kernel(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *y, unsigned char *u, unsigned char *v, int img_size)
{
    unsigned char R, G, B;
    unsigned char Y, cb, cr;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < img_size)
    {
        R = r[i];
        G = g[i];
        B = b[i];
        
        Y  = (unsigned char)( 0.299 * R + 0.587 * G +  0.114 * B);
        cb = (unsigned char)(-0.169 * R - 0.331 * G +  0.499 * B + 128);
        cr = (unsigned char)( 0.499 * R - 0.418 * G - 0.0813 * B + 128);
        
        y[i] = Y;
        u[i] = cb;
        v[i] = cr;
    }
}

__global__ void GPU_rgb2hsl_kernel(unsigned char *r, unsigned char *g, unsigned char *b, float *h, float *s, unsigned char *l, int img_size)
{
    float H, S, L;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < img_size)
    {  
        float var_r = ((float)r[i]/255);//Convert RGB to [0,1]
        float var_g = ((float)g[i]/255);
        float var_b = ((float)b[i]/255);
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max )
                {
                    H = (1.0/3.0) + del_r - del_b;
                }
                else
                {
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        h[i] = H;
        s[i] = S;
        l[i] = (unsigned char)(L * 255);
    }
}

__global__ void GPU_hsl2rgb_kernel(unsigned char *r, unsigned char *g, unsigned char *b, float *h, float *s, unsigned char *l, int img_size)
{
    unsigned char R, G, B;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < img_size)
    {
        float H = h[i];
        float S = s[i];
        float L = l[i]/255.0f;
        float var_1, var_2;
        
        
        if ( S == 0 )
        {
            R = L * 255;
            G = L * 255;
            B = L * 255;
        }
        else
        {            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            R = 255 * GPU_Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            G = 255 * GPU_Hue_2_RGB( var_1, var_2, H );
            B = 255 * GPU_Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        r[i] = R;
        g[i] = G;
        b[i] = B;
    }
}

PGM_IMG GPU_contrast_enhancement_gray(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *) malloc(result.w * result.h * sizeof(unsigned char));
    
    GPU_histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    GPU_histogram_equalization(result.img, img_in.img, hist, result.w * result.h, 256);
    return result;
}

PPM_IMG GPU_contrast_enhancement_color_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
    unsigned char * y_equ;
    int hist[256];
    
    yuv_med = GPU_rgb2yuv(img_in);
    y_equ = (unsigned char *) malloc(yuv_med.h * yuv_med.w * sizeof(unsigned char));
    
    GPU_histogram(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);
    GPU_histogram_equalization(y_equ,yuv_med.img_y, hist,yuv_med.h * yuv_med.w, 256);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    result = GPU_yuv2rgb(yuv_med);

    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
    
    return result;
}

PPM_IMG GPU_contrast_enhancement_color_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    int hist[256];

    hsl_med = GPU_rgb2hsl(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));

    GPU_histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);
    GPU_histogram_equalization(l_equ, hsl_med.l,hist,hsl_med.width*hsl_med.height, 256);
    
    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = GPU_hsl2rgb(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    return result;
}


PPM_IMG GPU_hsl2rgb(HSL_IMG img_in)
{
    PPM_IMG result;
    unsigned char *r, *g, *b, *l;
    float *h, *s;
    int img_size = img_in.width * img_in.height;

    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    
    cudaMalloc(&r, sizeof(unsigned char) * img_size);
    cudaMalloc(&g, sizeof(unsigned char) * img_size);
    cudaMalloc(&b, sizeof(unsigned char) * img_size);
    cudaMalloc(&h, sizeof(float) * img_size);
    cudaMalloc(&s, sizeof(float) * img_size);
    cudaMalloc(&l, sizeof(unsigned char) * img_size);
    cudaMemcpy(h, img_in.h, sizeof(float) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(s, img_in.s, sizeof(float) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(l, img_in.l, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

    dim3 blockDim(MAX_THREADS);
    dim3 gridDim((img_size + blockDim.x - 1)/ blockDim.x);
    
    GPU_hsl2rgb_kernel<<<gridDim, blockDim>>>(r, g, b, h, s, l, img_size);

    cudaMemcpy(result.img_r, r, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(result.img_g, g, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(result.img_b, b, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   

    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(h);
    cudaFree(s);
    cudaFree(l);

    return result;
}

HSL_IMG GPU_rgb2hsl(PPM_IMG img_in)
{
    HSL_IMG img_out;
    unsigned char *r, *g, *b, *l;
    float *h, *s;
    int img_size = img_in.w * img_in.h;

    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_size * sizeof(float));
    img_out.s = (float *)malloc(img_size * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    
    cudaMalloc(&r, sizeof(unsigned char) * img_size);
    cudaMalloc(&g, sizeof(unsigned char) * img_size);
    cudaMalloc(&b, sizeof(unsigned char) * img_size);
    cudaMalloc(&h, sizeof(float) * img_size);
    cudaMalloc(&s, sizeof(float) * img_size);
    cudaMalloc(&l, sizeof(unsigned char) * img_size);
    cudaMemcpy(r, img_in.img_r, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g, img_in.img_g, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, img_in.img_b, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

    dim3 blockDim(MAX_THREADS);
    dim3 gridDim((img_size + blockDim.x - 1)/ blockDim.x);
    
    GPU_rgb2hsl_kernel<<<gridDim, blockDim>>>(r, g, b, h, s, l, img_size);

    cudaMemcpy(img_out.h, h, sizeof(float) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(img_out.s, s, sizeof(float) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(img_out.l, l, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   

    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(h);
    cudaFree(s);
    cudaFree(l);
    
    return img_out;
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG GPU_rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    unsigned char *r, *g, *b, *y, *u, *v;
    int img_size = img_in.w * img_in.h;

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char) * img_size);

    cudaMalloc(&r, sizeof(unsigned char) * img_size);
    cudaMalloc(&g, sizeof(unsigned char) * img_size);
    cudaMalloc(&b, sizeof(unsigned char) * img_size);
    cudaMalloc(&y, sizeof(unsigned char) * img_size);
    cudaMalloc(&u, sizeof(unsigned char) * img_size);
    cudaMalloc(&v, sizeof(unsigned char) * img_size);
    cudaMemcpy(r, img_in.img_r, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g, img_in.img_g, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, img_in.img_b, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

    dim3 blockDim(MAX_THREADS);
    dim3 gridDim((img_size + blockDim.x - 1)/ blockDim.x);
    
    GPU_rgb2yuv_kernel<<<gridDim, blockDim>>>(r, g, b, y, u, v, img_size);

    cudaMemcpy(img_out.img_y, y, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(img_out.img_u, u, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(img_out.img_v, v, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   

    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);

    return img_out;
}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG GPU_yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    unsigned char *r, *g, *b, *y, *u, *v;
    int img_size = img_in.w * img_in.h;

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char) * img_size);

    cudaMalloc(&r, sizeof(unsigned char) * img_size);
    cudaMalloc(&g, sizeof(unsigned char) * img_size);
    cudaMalloc(&b, sizeof(unsigned char) * img_size);
    cudaMalloc(&y, sizeof(unsigned char) * img_size);
    cudaMalloc(&u, sizeof(unsigned char) * img_size);
    cudaMalloc(&v, sizeof(unsigned char) * img_size);
    cudaMemcpy(y, img_in.img_y, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(u, img_in.img_u, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(v, img_in.img_v, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

    dim3 blockDim(MAX_THREADS);
    dim3 gridDim((img_size + blockDim.x - 1)/ blockDim.x);
    
    GPU_yuv2rgb_kernel<<<gridDim, blockDim>>>(r, g, b, y, u, v, img_size);

    cudaMemcpy(img_out.img_r, r, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(img_out.img_g, g, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   
    cudaMemcpy(img_out.img_b, b, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);   

    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);
    
    return img_out;
}