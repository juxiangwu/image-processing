/* Please Write the OpenCL Kernel(s) code here*/
__kernel void static_tv_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float t = 1.0f;
   
   float r = rand(coord - t * t);
   float g = rand(coord - t * t * t);
   float b = rand(coord - t * t * t * t);
   
   float mx = max(max(r,g),b);
   
   float4 rgba = mix(color,(float4)(mx,mx,mx,1.0f),0.35f);
   
   write_imagef(output,convert_int2(coord),rgba);
}
