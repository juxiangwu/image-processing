/* Please Write the OpenCL Kernel(s) code here*/

__kernel void middle_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float max_val = max(color.x,max(color.y,color.z));
   float min_val = min(color.x,min(color.y,color.z));
   
   color.x = color.y = color.z = (max_val + min_val) / 2.0f;
   
   write_imagef(output,convert_int2(coord),color);
}