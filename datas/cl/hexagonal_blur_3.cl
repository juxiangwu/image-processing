/* Please Write the OpenCL Kernel(s) code here*/
__kernel void hexagonal_blur_3(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float blurSize = 256.0f;
   float imageWidth = size.x;
   float imageHeight = size.y;
   
   float2 offset = (float2)(blurSize/imageWidth, blurSize/imageHeight);
   
   float4 color2,color1,color3;
   
   for(int i = 0; i < 30; i++){
      color1 += 1.0f/30.0f * read_imagef(input,sampler, coord - convert_int2((float2)( offset.x * i, offset.y * i)));
   }
   
   for(int i = 0; i < 30; i++){
      color2 += 1.0f/30.0f * read_imagef(input,sampler, coord + convert_int2((float2)( offset.x * i, -offset.y * i)));
   }
   
   for(int i = 0; i < 30; i++){
      color3 += 1.0f/30.0f * read_imagef(input,sampler, coord + convert_int2((float2)( offset.x * i, -offset.y * i)));
   }
   
   
   write_imagef(output,coord,(color1 + color2 + color3) / 3);
}