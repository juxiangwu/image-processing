/* Please Write the OpenCL Kernel(s) code here*/

__kernel void relief_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float4 color2 = read_imagef(input,sampler,coord + (int2)(1,1));
   
   float4 dst_color = color2 - color + 128.0f / 255.0f;
   
   write_imagef(output,coord,dst_color);
   
}