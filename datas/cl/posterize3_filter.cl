/* Please Write the OpenCL Kernel(s) code here*/

__kernel void posterize3_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int level = 1;
   int colors = 40;
   float numAreas = 255.0f / level;
   float numValues = 255.0f / (level - 1);
   
   local float levels[256];
   
   for(int i = 0;i < 256;i++){
       if(i < colors * level){
           levels[i] = colors * (level - 1) / 255.0f;
       }else {
           levels[i] = colors * level / 255.0f;
           ++level;
       }
   }
   
   float3 src_rgb = read_imagef(input,sampler,coord).xyz;
   
   float3 dst_rgb;
   dst_rgb.x = levels[convert_int(src_rgb.x * 255)];
   dst_rgb.y = levels[convert_int(src_rgb.y * 255)];
   dst_rgb.z = levels[convert_int(src_rgb.z * 255)];
   
   write_imagef(output,coord,(float4)(dst_rgb,1.0f));
}