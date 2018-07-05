/* Please Write the OpenCL Kernel(s) code here*/
__kernel void mcgreen_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float3 ink = (float3)(0.32f,0.50f,0.0f);
   float3 c11 = read_imagef(input,sampler,coord).xyz;
   float3 mcgreen = (float3)(0.0f,1.0f,1.0f);
   float3 lct = floor(mcgreen * length(c11)) / mcgreen;
   
   write_imagef(output,convert_int2(coord),(float4)(lct * ink,1.0f));
}
