/* Please Write the OpenCL Kernel(s) code here*/

float luminance(float4 color){
    const float3 w = (float3)(0.2125, 0.7154, 0.0721);
    return dot(color.xyz, w);
}
__kernel void duotone_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   float l = luminance(color);
   float e = 0.0f;
   float3 highlight = (float3)(1.0f,0.0f,0.0f);
   float3 shadow = (float3)(0.5f,0.5f,0.5f);
   
   float3 h = (highlight + e) / (luminance((float4)(highlight,1.0f)) + e) * l;
   
   float3 s = (shadow + e) / (luminance((float4)(shadow,1.0f)) + e) * l;
   
   float3 c = h * l + s * (1.0f - l);
   
   write_imagef(output,convert_int2(coord),color);
}