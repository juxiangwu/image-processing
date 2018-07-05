/* Please Write the OpenCL Kernel(s) code here*/
__kernel void clamp_filter(__read_only image2d_t input,__write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                               
    const int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float4 color = read_imagef(input,sampler,coord);
    float factor = 1.5f;
    float v = 0.5f;
    color = color * factor + 0.5f;
    
    if(color.x > 1.0f){
        color.x = 1.0f;
    } 
    
    if(color.y > 1.0f){
        color.y = 1.0f;
    }
    
    if(color.z > 1.0f){
        color.z = 1.0f;
    }
    
    color.w = 1.0f;
    
    write_imagef(output,coord,color);
}