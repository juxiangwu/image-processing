/* Please Write the OpenCL Kernel(s) code here*/
__kernel void video_effect_3x3_filter(__read_only image2d_t input,__write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                               
    const int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int sPattern_3x3[9] = {0, 1, 2, 2, 0, 1,1, 2, 0,};
    
    int pattern_width  = 3;
    int pattern_height = 3;
    
    float3 src_rgb = read_imagef(input,sampler,coord).xyz * 255.0f;
    float3 dst_rgb = src_rgb;
    
    int index =  pattern_width * (coord.y % pattern_height) + (coord.x % pattern_width);
    
    switch(index){
    case 0:
        dst_rgb.x = clamp(src_rgb.x * 2.0f,0.0f,255.0f);
    break;
    case 1:
        dst_rgb.y = clamp(src_rgb.y * 2.0f,0.0f,255.0f);
    break;
    case 3:
        dst_rgb.z = clamp(src_rgb.z * 2.0f,0.0f,255.0f);
    break;
    }
    
    write_imagef(output,coord,(float4)(dst_rgb / 255.0f,1.0f));
}