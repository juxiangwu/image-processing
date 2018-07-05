/* Please Write the OpenCL Kernel(s) code here*/

__kernel void emboss_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
     
    const int maskrows = 3 / 2;
    const int maskcols = 3 / 2;
    float emboss_mask[9] = {2,0,0,0,-1,0,0,0,-1};
   float3 color = (float3)(0,0,0);
   int idx = 0;
  
    for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float3 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y)).xyz;
        color += srcColor * emboss_mask[idx];
        idx++;
      }
    }
    
   float gray = (color.x + color.y + color.z) / 3.0f;
   int use_gray = 1;
   if(!use_gray){
       write_imagef(output,coord,(float4)(color,1.0f));
   }else{
       write_imagef(output,coord,(float4)(gray,gray,gray,1.0f));
   }
    
}

