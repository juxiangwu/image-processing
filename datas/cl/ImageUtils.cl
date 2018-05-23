
__kernel void white_balance(__global const float* lms, __global float* result, __global const int* image_shape, __global const float* factors, __global const float* white)
{
    float r, g, b, x, y, z;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = i * h * c  + j * c;

	result[idx] = (lms[idx] - factors[0])/(factors[1]- factors[0]) * white[0];
	result[idx+1] = (lms[idx+1] - factors[2])/(factors[3]- factors[2]) * white[1];
	result[idx+2] = (lms[idx+2] - factors[4])/(factors[5]- factors[4]) * white[2];
}

__kernel void sample_image(__global const float* image, __global float* sampledImage, __global const int* image_shape, __global const int* sampled_shape)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int h = sampled_shape[1];
    int c = sampled_shape[2];
    int idx = x * h * c  + y * c;

    int originalX = (int)(0.5f+(float)x*(float)(image_shape[0]-1)/(float)(sampled_shape[0]-1));
    int originalY = (int)(0.5f+(float)y*(float)(image_shape[1]-1)/(float)(sampled_shape[1]-1));

    int idx_original = originalX * image_shape[1] * c + originalY * c;

    sampledImage[idx] = image[idx_original];
    sampledImage[idx+1] = image[idx_original+1];
    sampledImage[idx+2] = image[idx_original+2];
}

__kernel void low_pass_x(__global const float* image, __global float* smoothed, __global const float* gauss, __global const int* image_shape, __global const int* gauss_shape)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = image_shape[0];
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = x * h * c  + y * c;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    float sum = 0.0f, coeff=0.0f;

    int k = (int) (gauss_shape[0]-1)/2;
    int start = x - k;
    int end = x + k;
    int j = 0;

    //float coeffs[] = {0.0545f, 0.2442f, 0.4026f, 0.2442f, 0.0545f};

    for(int i = start; i <= end; i++) {
        coeff = gauss[j];
        if( (i >= 0) && (i < w)) {
            r += image[i*h*c + y*c]*coeff;
            g += image[i*h*c + y*c + 1]*coeff;
            b += image[i*h*c + y*c + 2]*coeff;
            sum += coeff;
        }
        j++;
    }

    smoothed[idx] = r/sum;
    smoothed[idx+1] = g/sum;
    smoothed[idx+2] = b/sum;
}

__kernel void low_pass_y(__global const float* image, __global float* smoothed, __global const float* gauss, __global const int* image_shape, __global const int* gauss_shape)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = image_shape[0];
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = x * h * c  + y * c;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    float sum = 0.0f, coeff=0.0f;

    int k = (int) (gauss_shape[0]-1)/2;
    int start = y - k;
    int end = y + k;
    int j = 0;

    //float coeffs[] = {0.0545f, 0.2442f, 0.4026f, 0.2442f, 0.0545f};

    for(int i = start; i <= end; i++) {
        coeff = gauss[j];
        if( (i >= 0) && (i < h)) {
            r += image[x*h*c + i*c]*coeff;
            g += image[x*h*c + i*c + 1]*coeff;
            b += image[x*h*c + i*c + 2]*coeff;
            sum += coeff;
        }
        j++;
    }

    smoothed[idx] = r/sum;
    smoothed[idx+1] = g/sum;
    smoothed[idx+2] = b/sum;
}

__kernel void high_pass(__global const float* image, __global const float* smoothed, __global float* sharpened, __global const int* image_shape)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int h = image_shape[1];
    int c = image_shape[2];
    int idx = x * h * c  + y * c;

    //sharpened[idx] = 2.0f*image[idx]-smoothed[idx];
    //sharpened[idx+1] = 2.0f*image[idx+1]-smoothed[idx+1];
    sharpened[idx] = image[idx];
    sharpened[idx+1] = image[idx+1];
    sharpened[idx+2] = 2.0f*image[idx+2]-smoothed[idx+2];
}