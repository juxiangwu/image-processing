/* Please Write the OpenCL Kernel(s) code here*/

__kernel void pencil_sketch_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                              
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float3 src_rgb = read_imagef(input,sampler,coord).xyz;
   float3 invert_rgb = 1.0f - src_rgb;
   
   float3 mask [9] = {0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.1f};
   
   const int maskrows = 3 / 2;
   const int maskcols = 3 / 2;

   float4 color = (float4)(0,0,0,1.0f);
   int idx = 0;

   for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));
          color.xyz += srcColor.xyz * mask[idx];
        idx++;
      }
   }
   color.xyz = color.xyz / 9;
   
   float3 dst_rgb = clamp(invert_rgb + color.xyz,0,1);
  // dst_rgb.x = dst_rgb.y = dst_rgb.z = (dst_rgb.x + dst_rgb.y + dst_rgb.z) / 3;
   write_imagef(output,coord,(float4)(dst_rgb,1.0f));
}