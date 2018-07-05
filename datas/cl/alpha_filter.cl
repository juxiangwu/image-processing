/* Please Write the OpenCL Kernel(s) code here*/

__kernel void alpha_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float factor = -0.5f;

    float4 color = read_imagef(input,sampler,coord);
    color.w = clamp(factor,0.0f,1.0f);
    write_imagef(output,coord,color);
}