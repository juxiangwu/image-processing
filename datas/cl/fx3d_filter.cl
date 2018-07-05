/* Please Write the OpenCL Kernel(s) code here*/
float mod2(float x,float y){
    return x - y * floor(x / y);
}

__kernel void fx3d_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float4 color = read_imagef(input,sampler,coord);
   float gammaed = 0.15f;
   
   float leifx_linegamma = gammaed;
   float2 res;
   res.x = size.x;
   res.y = size.y;
   
   float2 dithet = coord.xy * res.xy;
   
   dithet.y = coord.y * res.y;
   
   float horzline1 = (mod2(dithet.y,2.0f));
   
   if(horzline1 < 1.0f){
       leifx_linegamma = 0.0f;
   }
   
   float leifx_gamma = 1.3f - gammaed + leifx_linegamma;
   
   float4 rgba = pow(color,1.0f / leifx_gamma);
   
   rgba.w = 1.0f;
   
   write_imagef(output,convert_int2(coord),rgba);
}
