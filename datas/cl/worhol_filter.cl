/* Please Write the OpenCL Kernel(s) code here*/
__kernel void worhol_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float steps = 2.0f;
   float dotsize = 1.0f / steps;
   float half_step = dotsize / 2.0f;
   
   float2 coord2 = coord * steps;
   
   float4 color = read_imagef(input,sampler,coord2);
   
   float4 tint;
   
   float ofs = coord.x * steps + coord.y * steps * 2;
   
   if(ofs == 0.0f){
       tint = (float4)(1.0f,1.0f,0.0f,0.0f);
   }else if(ofs == 1.0f){
       tint = (float4)(0.0f,0.0f,1.0f,0.0f);
   }else{
       tint = (float4)(0.0f,1.0f,1.0f,0.0f);
   }
   
   float gray = dot(color.xyz,(float3)(0.3f,0.59f,0.11f));
   
   float4 dst_color = mix(color,tint,gray);
   
   write_imagef(output,convert_int2(coord),dst_color);
}