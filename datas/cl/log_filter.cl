
__kernel void log_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));

     float4 color;

     float4 srcColor = read_imagef(input,sampler,coord);

     float3 logval = log(1.0f + srcColor.xyz);
     float scale = 2.0f;
     color = scale * (float4)(logval,1.0f);

     write_imagef(output,coord,color);
}

/*
*  y = a + log(1 + x) / b 
*/
__kernel void log_2_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));

   
     float4 color = read_imagef(input,sampler,coord);
    
     float a = 0.0f;
     float b = 0.025f;
     
     float4 dst_color = a + log(1.0f + color) / b;
     dst_color.w = 1.0f;
     write_imagef(output,coord,dst_color);
}
