/* Please Write the OpenCL Kernel(s) code here*/
__kernel void denoise_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 center_color = read_imagef(input,sampler,coord);
   
   float exponent = 0.5f;
   float strength = 1.0f;
   float total = 0.0f;
   float4 color;
   for (int x = -4; x <= 4; x += 1) {
     for (int y = -4; y <= 4; y += 1) {
           float4 sample = read_imagef(input,sampler, (coord + (int2)(x, y)));
           float dot_res = dot(sample.xyz - center_color.xyz, (float3)(0.25,0.25,0.25));
           float weight = 1.0f - ((dot_res > 0)? dot_res : -dot_res);
           weight = pow(weight, exponent);
           color += sample * weight;
           total += weight;
     }
   }
   
  write_imagef(output,coord,color / total);
}