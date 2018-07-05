/* Please Write the OpenCL Kernel(s) code here*/
void filter2d_internal(__read_only image2d_t input,
                       __write_only image2d_t output,
                       const int maskWidth,
                       const int maskHeight,
                        float * mask,int compute_aver){

   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));

   const int maskrows = maskWidth / 2;
   const int maskcols = maskHeight / 2;

   float4 color = (float4)(0,0,0,1.0f);
   int idx = 0;

   for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));
          color.xyz += srcColor.xyz * mask[idx];
        idx++;
      }
   }
   if(compute_aver){
     color.xyz = color.xyz / (maskWidth * maskHeight);
   }
  write_imagef(output,coord,color);
}

__kernel void bump_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
  float color_matrix[9] = {-1.0f,-1.0f,0.0f,
                           -1.0f,1.0f,1.0f,
                           0.0f,1.0f,1.0f};
  filter2d_internal(input,output,3,3,color_matrix,0);
}
