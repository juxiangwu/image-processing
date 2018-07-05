/* Please Write the OpenCL Kernel(s) code here*/
__kernel void contrast_filter_3(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));
     
     float arg = 1.5f;
     float4 color = read_imagef(input,sampler,coord);
     float slope = arg > 0.5f ? 1.0f/(2.0f - 2.0f * arg) : 2.0f * arg;
     float4 dstcolor = (float4)((color.xyz-0.5f)*slope+0.5f, color.w);
     
     write_imagef(output,coord,dstcolor);
}