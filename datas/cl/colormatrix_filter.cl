/* Please Write the OpenCL Kernel(s) code here*/
__kernel void colormatrix_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color_matrix = (float4)(0.0f,0.0f,1.0f,1.0f);
   float4 color = read_imagef(input,sampler,coord);
   float fade_const = 0.25f;
   
   float4 dst_color = color * (1.0f - fade_const) + fade_const * color * color_matrix;
   
   write_imagef(output,convert_int2(coord),dst_color);
}