/* Please Write the OpenCL Kernel(s) code here*/
__kernel void sepia_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float4 rgb = read_imagef(input,sampler,coord);
    
    float r = rgb.x,g = rgb.y,b = rgb.z;
    
    rgb.x = (r * 0.393f) + (g * 0.769f) + (b * 0.189f);
    rgb.y = (r * 0.349f) + (g * 0.686f) + (b * 0.168f);
    rgb.z = (r * 0.272f) + (g * 0.534f) + (b * 0.131f);
    
    
    write_imagef(output,coord,rgb);
}