/* Please Write the OpenCL Kernel(s) code here*/
__kernel void skin_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   float4 color = read_imagef(input,sampler,coord);
   
   float r = color.x;
   float g = color.y;
   float b = color.z;
   
   float xy = (r - g);
   xy = xy > 0 ? xy : -xy;
   
   if((r <= 45.0f / 255.0f) || 
      (g <= 40.0f / 255.0f) || 
      (b <= 20.0f / 255.0f) ||
      (r <= g) ||
      (r <= b) ||
      ((r - min(g,b)) <= 15.0f / 255.0f) ||
      (xy <= 15.0f / 255.0f)){
      color.x = color.y = color.z = 0;
   }
   write_imagef(output,convert_int2(coord),color);
}