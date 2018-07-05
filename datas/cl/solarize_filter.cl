/* Please Write the OpenCL Kernel(s) code here*/
__kernel void solarize_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float amount = 0.25f;
   
   float4 rgb = read_imagef(input,sampler,coord);
   
   if(rgb.x > amount) rgb.x = 1.0 - rgb.x;
   if(rgb.y > amount) rgb.y = 1.0 - rgb.y;
   if(rgb.z > amount) rgb.z = 1.0 - rgb.z;
   
   write_imagef(output,coord,rgb);

}
