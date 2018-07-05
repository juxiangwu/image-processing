/* Please Write the OpenCL Kernel(s) code here*/

__kernel void casting_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   float3 rgb;
   
   float pixel = color.x * 128.0f / (color.y + color.z + 1);
   if(pixel < 0){
       pixel = 0;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.x = pixel;
   
   pixel = color.y * 128.0f / (color.x + color.z + 1);
   if(pixel < 0){
       pixel = 0;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.y = pixel;
   
   pixel = color.z * 128.0f / (color.x + color.y + 1);
   if(pixel < 0){
       pixel = 0;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.z = pixel;
   rgb /= 255.0f;
   write_imagef(output,coord,(float4)(rgb, 1.0f));
   
}