/* Please Write the OpenCL Kernel(s) code here*/
__kernel void oil_paint2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int R = 16;
   int xLength = 2 * R + 1;
   
   float4 color = read_imagef(input,sampler,coord) * 255;
   float gray  = (color.x + color.y + color.z) / 3;
   
   float every = (gray / R) * R;
   
   color.x = color.y = color.z = every;
   
   write_imagef(output,coord,color / 255);
}