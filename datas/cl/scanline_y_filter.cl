/* Please Write the OpenCL Kernel(s) code here*/
__kernel void scanline_y_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                             
  
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int2 resolution = {size.x,size.y};
    float scale = 1.0f;
    if(fmod(floor((float)(coord.y) / scale),3.0f) == 0.0f){
        write_imagef(output,coord,(float4)(0.0f,0.0f,0.0f,1.0f));    
    }else{
        float4 color = read_imagef(input,sampler,coord);
        write_imagef(output,coord,color);
    }
    
}