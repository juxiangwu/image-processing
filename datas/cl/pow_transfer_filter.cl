/* Please Write the OpenCL Kernel(s) code here*/
/*
* y = c * pow(x / 255.0f,r) + b
*/
__kernel void pow_transfer_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));

   
     float4 color = read_imagef(input,sampler,coord) * 255.0f;
    
     float c = 1.0f;
     float b = 128.0f / 255.0f;
     float r = 3.0f;
     
     float4 dst_color = c * pow(color / 255.0f, r) + b;
     //dst_color /= 255.0f;
     dst_color.w = 1.0f;
     write_imagef(output,coord,dst_color);
}
