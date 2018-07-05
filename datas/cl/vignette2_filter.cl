/* Please Write the OpenCL Kernel(s) code here*/
__kernel void vignette2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 lensRadius = (float2)(0.80f,0.40f);
   
   float4 rgba = read_imagef(input,sampler,coord);
   
   float d = distance(1.0f / coord,(float2)(0.5f,0.5f));
   
   rgba *= smoothstep(lensRadius.x,lensRadius.y,d);
 //  rgba.w = 1.0f;
   write_imagef(output,convert_int2(coord),rgba);
   
}