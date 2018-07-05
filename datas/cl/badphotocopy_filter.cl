/* Please Write the OpenCL Kernel(s) code here*/

__kernel void badphotocopy_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float noise = rand(color.xy) / 2.0f;
   
   float avg = (length(color.xyz) / 3.0f) * 0.75f + noise * 0.25f;
   
   if(avg > 0.25f){
       avg = 1.0f;
   }else{
       avg = 0.0f;
   }
   color.xyz = avg;
   write_imagef(output,coord,color);
   
}