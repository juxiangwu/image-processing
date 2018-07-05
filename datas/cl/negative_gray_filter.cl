/* Please Write the OpenCL Kernel(s) code here*/

__kernel void negative_gray_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float gray = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
   gray = 1.0f - gray;
   
   color.x = color.y = color.z = gray;
   
   write_imagef(output,coord,color);
   
}