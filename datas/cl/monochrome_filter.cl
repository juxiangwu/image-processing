/* Please Write the OpenCL Kernel(s) code here*/
__kernel void monochrome_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float arg = 0.75f;
   float4 color = read_imagef(input,sampler,coord);
   float y = dot(color.xyz, (float3)(0.299f, 0.587f, 0.114f));
   float4 dst_color = (float4)(mix(color.xyz, (float3)(y), arg), 1.0f);
   
   write_imagef(output,convert_int2(coord),dst_color);
}