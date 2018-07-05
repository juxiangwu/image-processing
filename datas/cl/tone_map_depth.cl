/* Please Write the OpenCL Kernel(s) code here*/

float3 tone_map(float3 hdrRGB,float exposure){
    
    float3 dstRGB = 1.0f - exp2(-hdrRGB * exposure);
    return dstRGB;
    
}


__kernel void tone_map_depth(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float exposure = 2.5f;
    
    float4 color = read_imagef(input,sampler,coord);
    
    float4 rgba = 1.0f - exp2(-color * exposure);
    rgba.w = 1.0f;
    
    write_imagef(output,coord,rgba);
}