/* Please Write the OpenCL Kernel(s) code here*/

__kernel void thermal_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float4 color = read_imagef(input,sampler,coord);
   float4 colors[3] = {(float4)(0.0f,0.0f,1.0f,1.0f),(float4)(1.0f,1.0f,0.0f,1.0f),(float4)(1.0f,0.0f,0.0f,1.0f)};
   
   float lum = dot((float3)(0.30f,0.59f,0.11f),color.xyz);
   int ix = (lum < 0.5f) ?  0.0f : 1.0f;
   
   float4 thermal = mix(colors[ix],colors[ix + 1],(lum - (float)ix * 0.5f) / 0.5f);
   
   write_imagef(output,convert_int2(coord),thermal);
   
}