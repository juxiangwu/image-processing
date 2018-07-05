/* Please Write the OpenCL Kernel(s) code here*/
__kernel void adjust_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float3 rgb = read_imagef(input,sampler,coord).xyz;
    
    float r = 0.0f;
    float g = 0.2f;
    float b = 0.0f;
    
    rgb.x += r;
    rgb.y += g;
    rgb.z += b;
    
    
    write_imagef(output,coord,(float4)(rgb,1.0f));
}