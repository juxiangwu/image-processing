/* Please Write the OpenCL Kernel(s) code here*/
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

   
     float4 color = read_imagef(input,sampler,coord) * 255.0f;
    
     float a = 0.0f;
     float b = 0.015f;
     
     float4 dst_color = a + log(1.0f + color) / b;
     dst_color /= 255.0f;
     dst_color.w = 1.0f;
     write_imagef(output,coord,dst_color);
}