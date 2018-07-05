/* Please Write the OpenCL Kernel(s) code here*/
__kernel void contrast_filter_2(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));

     float4 color;

     float4 srcColor = read_imagef(input,sampler,coord);
     float scale = 2.0f;
     float3 contrast = (exp(2 * (srcColor.xyz - 0.5f) * scale) - 1) / (exp(2 * (srcColor.xyz - 0.5f) * scale) + 1);

     color = scale * (float4)(contrast,1.0f);

     write_imagef(output,coord,color);
}
