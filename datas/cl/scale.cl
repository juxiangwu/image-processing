/* Please Write the OpenCL Kernel(s) code here*/

__kernel void scale(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int width = size.x;
    int height = size.y;
    
    float scale = 0.25f;
    float scalex = 1.0f / (scale * width);
    float scaley = 1.0f / (scale * height);
    
    float2 scale_coord = convert_float2(coord) * (float2)(scalex,scaley);
    
    float4 color = read_imagef(input,sampler,scale_coord);
    
    write_imagef(output,coord,color);
}