/* Please Write the OpenCL Kernel(s) code here*/
__kernel void tan_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_TRUE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   
   float scale = 2.0f;
   float2 new_coord =tan(coord * scale);
   float4 color = read_imagef(input,sampler,new_coord);
   
   write_imagef(output,convert_int2(coord),color);
   
}