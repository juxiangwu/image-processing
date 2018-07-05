/* Please Write the OpenCL Kernel(s) code here*/
float lum(float3 color){
    return 0.3f * color.x + 0.59f * color.y + 0.11f * color.z;
}

__kernel void color_clip_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float L = lum(color.xyz);
   
   float n = min(min(color.x,color.y),color.z);
   float x = max(max(color.x,color.y),color.z);
   
   if(n < 0.0f){
      color.xyz = L +(((color.xyz - L) * L) / (L - n));
   }
   
   if(x > 1.0f){
      color.xyz = L + (((color.xyz - L) * (1 - L)) / (x - L));
   }
   
   color.xyz += (0.5f - L);
   
   write_imagef(output,convert_int2(coord),color);
   
}