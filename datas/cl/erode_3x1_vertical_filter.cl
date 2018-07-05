/* Please Write the OpenCL Kernel(s) code here*/
__kernel void erode_3x1_vertical_filter(__read_only image2d_t input,__write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                              
   const int2 size = get_image_dim(input);
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float grays[3] = {0.0f};
   float thresh = 80.0f / 255.0f;
   int idx = 0;
   float dstgray = 0.0f;
   int hasZero = 0;
   for(int i = -1;i <= 1;i++){
       float3 rgb = read_imagef(input,sampler,(int2)(coord.x ,coord.y + i)).xyz;
       float gray = (rgb.x + rgb.y + rgb.z) / 3.0f;
       if(gray >= thresh){
           gray = 1.0f;
       }else{
           gray = 0.0f;
           hasZero = 1;
       }
       grays[idx] = gray;
       dstgray += gray;
       idx++;
   }
   
   dstgray = dstgray / 3.0f;
   
   
   float4 dstrgb = (float4)(dstgray,dstgray,dstgray,1.0f);
   
   if(hasZero){
        dstrgb.x = dstrgb.y = dstrgb.z = 0.0f;   
   }
   
   write_imagef(output,coord,dstrgb);
   
}