/* Please Write the OpenCL Kernel(s) code here*/

float gaussian_dim1d_blur_compute(int i,float sigma){
    float sigmaq = sigma * sigma;
    float value = 0.0f;
    value = exp(-((i * i) / (2.0f * sigmaq))) / sqrt(2.0f * 3.14159265358979323846f * sigmaq);
    return value;
}

__kernel void vert_horiz_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

   
    float weight[31] = {0.0f};
    float sum = 0.0f;
    float sigma = 15.0f;
    
    for(int i = 1;i <= 31;i++){
        weight[i - 1] = gaussian_dim1d_blur_compute(i,sigma);
        sum += 2.0f * weight[i - 1];
    }
    
    for(int i = 0;i < 31;i++){
        weight[i] = weight[i] / sum;
    }

    
    float3 dstColor;
    dstColor = read_imagef(input,sampler,coord).xyz * weight[0];
    for(int i = 0;i < 31;i++){
       // if(coord.x != 0){
            int2 offset = (int2)(i ,i);
            dstColor += read_imagef(input,sampler,coord - offset).xyz * weight[i];
            dstColor += read_imagef(input,sampler,coord + offset).xyz * weight[i];
       // }
    }
    
    float4 color = (float4)(dstColor,1.0f);
    
    write_imagef(output,coord,color);
}