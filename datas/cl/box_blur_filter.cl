/* Please Write the OpenCL Kernel(s) code here*/
__kernel void box_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float blurSize =256.0f; 
   float imageWidth = size.x;
   
   float4 sum;
   
   for(int i = 0; i < 40; i++){
      sum += read_imagef(input,sampler, coord + (int2)(0, convert_int((i-20) * blurSize / imageWidth)));
   }
  /*
   for(int i = 0; i < 40; i++){
      sum += read_imagef(input,sampler, coord + (int2)(convert_int((i-20) * blurSize / size.y),0));
   }
   */
   sum = sum / 40;
   
   write_imagef(output,coord,sum);
}