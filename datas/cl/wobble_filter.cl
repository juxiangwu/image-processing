/* Please Write the OpenCL Kernel(s) code here*/
__kernel void wobble_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 offset = (float2)(5.0f,5.0f);
   float2 freq = (float2)(5,5);
   float2 strength = (float2)(0.02f,0.02f);
   float time = 10000.0f;
   float2 tex_coord;
   tex_coord.x = coord.x + sin(coord.y * freq.x * time / 10000.0f + offset.x) * strength.x;
   tex_coord.y = coord.y + sin(coord.x * freq.y * time / 10000.0f + offset.y) * strength.y;
   
   float4 color = read_imagef(input,sampler,tex_coord);
   write_imagef(output,convert_int2(coord),color);
   
}