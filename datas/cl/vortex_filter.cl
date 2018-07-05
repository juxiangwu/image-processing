/* Please Write the OpenCL Kernel(s) code here*/
__kernel void vortex_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 uy;
   float2 resolution = (float2)(0.035f,0.035f);
   float2 p = -1.0f * convert_float2(dim) + 2.0f * coord /  resolution;
   float time = 1.0f;
   float a = atan2(p.y,p.x);
   float r = sqrt(dot(p,p));
   float s = r * (1.0f + 0.8f * cos ( time * 1.0f));
   
   uy.x = 0.02f * p.x + 0.03 * cos(-time + a * 3.0f) / s;
   uy.y = 0.1f * time + 0.02 * p.y + 0.03 * sin(-time + a * 3.0f) / s;
   
   float w = 0.9f + pow(max(1.5f - r,0.0f),4.0f);
   w *= 0.7f + 0.3f * cos(time + 3.0f * a);
   
   float4 color = read_imagef(input,sampler,uy);
   color.xyz = w * color.xyz;
   
   write_imagef(output,convert_int2(coord),color);
   
}
