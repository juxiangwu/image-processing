/* Please Write the OpenCL Kernel(s) code here*/

__kernel void split_blue_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float amount = 0.25f;
   
   float4 rgb = read_imagef(input,sampler,coord);
   
   rgb.xy = 0;
   
   write_imagef(output,coord,rgb);

}