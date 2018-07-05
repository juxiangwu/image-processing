/* Please Write the OpenCL Kernel(s) code here*/
__kernel void quilez_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 p = coord;
   
   p = p * convert_float2(size) + (float2)(convert_float2(size) / 2);
   
   float2 i = floor(p);
   float2 f = p - i;
   
   f = f * f * f * (f * (f * 6.0f - (float2)(15.0f,15.0f)) + (float2)(10.0f,10.0f));
   
   p = i + f;
   p = (p - (float2)(convert_float2(size / 2))) / convert_float2(size);
   
   float4 color = read_imagef(input,sampler,p);
   
   write_imagef(output,convert_int2(coord),color);
}