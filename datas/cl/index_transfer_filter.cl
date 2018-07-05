/* Please Write the OpenCL Kernel(s) code here*/
__kernel void index_transfer_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));

   
     float4 color = read_imagef(input,sampler,coord) * 255.0f;
    
     float a = 0.0f;
     float c = 0.065f;
     float b = 1.5f;
     
     
     float4 dst_color = pow(b, c  * (color - a)) - 1.0f;
     dst_color /= 255.0f;
     dst_color.w = 1.0f;
     write_imagef(output,coord,dst_color);
}