/* Please Write the OpenCL Kernel(s) code here*/
__kernel void linear_dot(__read_only image2d_t input,__write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                              
   const int2 size = get_image_dim(input);
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 colorSrc = read_imagef(input,sampler,coord);
   float a = 0.3f;
   float b = 0.3f;
   float3 rgb = (float3)(a * colorSrc.xyz + b);
   float4 colorDst = (float4)(rgb,1.0f);
   write_imagef(output,coord,colorDst);
}