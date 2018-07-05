/* Please Write the OpenCL Kernel(s) code here*/
__kernel void erode_7x7_cross_filter(__read_only image2d_t input,__write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   int2 coord = (int2)(get_global_id(0),get_global_id(1));


   float thresh = 80.0f / 255.0f;

   float grays[49] = {0.0f};
   int idx = 0;
   float dstgray = 0.0f;
   int hasZero = 0;
   for(int i = -3; i <= 3;i++){
       for(int j = -3;j <= 3;j++){
         float3 rgb = read_imagef(input,sampler,(int2)(coord.x + i,coord.y + j)).xyz;
         float gray = (rgb.x + rgb.y + rgb.z) / 3.0f;
         if(gray >= thresh){
            gray = 1.0f;
         }else{
            gray = 0.0f;
            hasZero = 1;
         }
        grays[idx] = gray;
        dstgray += gray;
        idx += 1;
      }
   }

   dstgray = dstgray / 49.0f;
   
   float4 dst_rgb = (float4)(dstgray,dstgray,dstgray,1.0f);

   if(hasZero == 1){
       dst_rgb = (float4)(0,0,0,1);
   }

   write_imagef(output,coord,dst_rgb);
}