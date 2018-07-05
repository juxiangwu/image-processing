/* Please Write the OpenCL Kernel(s) code here*/
__kernel void backto1980_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float avg = length(color.xyz) / 3.0f;
   float levels = 2.0f;
   avg = floor(avg * levels * 3.0f) / levels;
   
   color.x = avg;
   color.y = avg;
   color.z = avg;
   color.w = 1.0f;
   
   write_imagef(output,coord,color);
   
}