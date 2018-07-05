/* Please Write the OpenCL Kernel(s) code here*/
__kernel void below_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 replace_color = (float4)(1.0f,1.0f,1.0f,1.0f);
   float4 thresh = (float4)(0.4f,0.4f,0.4f,1.0f);
   
   float4 color = read_imagef(input,sampler,coord);
   
   if(color.x < thresh.x && color.y < thresh.y && color.z < thresh.z){
       color = replace_color;
   }
   
   write_imagef(output,convert_int2(coord),color);
   
}