/* Please Write the OpenCL Kernel(s) code here*/
__kernel void ice2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   float3 rgb;
   float pixel = color.x - color.y - color.z;
   pixel = pixel * 3.0f / 2.0f;
   
   if(pixel < 0){
       pixel = -pixel;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.x = pixel;
   
   pixel = color.y - color.x - color.z;
    pixel = pixel * 3.0f / 2.0f;
    if(pixel < 0){
       pixel = -pixel;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.y = pixel;
   
   pixel = color.z - color.x - color.y;
    pixel = pixel * 3.0f / 2.0f;
    if(pixel < 0){
       pixel = -pixel;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.z = pixel;
   
   rgb /= 255.0f;
   
   write_imagef(output,coord,(float4)(rgb,1.0f));
}