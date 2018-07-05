/* Please Write the OpenCL Kernel(s) code here*/

/*
  R = 0.393 * r + 0.769 * g + 0.189 * b
  G = 0.349 * r + 0.686 * g + 0.168 * b;
  B = 0.272 * r + 0.534 * g + 0.131 * b;
*/

__kernel void old_photo_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   
   float3 rgb;
   rgb.x = color.x * 0.393f + color.y * 0.769f + color.z * 0.189f;
   rgb.y = color.x * 0.249f + color.y * 0.686f + color.z * 0.168f;
   rgb.z = color.x * 0.272f + color.y * 0.534f + color.z * 0.131f;
   
   rgb /= 255.0f;
   
   write_imagef(output,coord,(float4)(rgb,1.0f));
 
}