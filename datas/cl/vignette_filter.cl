/* Please Write the OpenCL Kernel(s) code here*/
__kernel void vignette_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float amount = 2.0f;
   float2 vUv = (float2)(100,100);
   
   float dist = distance(vUv,(float2)(dim.x / 2,dim.y / 2));
   
   float4 color = read_imagef(input,sampler,coord);
   float size = 256.0f;
   color.xyz *= smoothstep(0.8f,size * 0.799f,dist * (amount + size));
   
   write_imagef(output,convert_int2(coord),color);
   
}