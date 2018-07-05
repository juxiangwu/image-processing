/* Please Write the OpenCL Kernel(s) code here*/
__kernel void hdtv_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float4 color = read_imagef(input,sampler,coord);
   float x = 4.0f;
   float r = pow(color.x,x) * 44403.0f / 200000.0f;
   float g = pow(color.y,x) * 141331.0f / 200000.0f;
   float b = pow(color.z,x) * 7133 / 100000.0f;
   
   float y = pow(r + g + b,1.0f / x);
   
   color.x = color.y = color.z = y;
   
   write_imagef(output,convert_int2(coord),color);
}