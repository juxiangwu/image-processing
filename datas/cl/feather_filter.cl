/* Please Write the OpenCL Kernel(s) code here*/
__kernel void feather_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    int width = size.x;
    int height = size.y;
    float ratio = (width > height ? width / height : height / width);
    
    int cx = size.x >> 1;
    int cy = size.y >> 1;
    
    float featherSize = 0.25f;
    int maxval = cx * cx + cy * cy;
    int minval = (int)(maxval * (1 - featherSize));
    
    int diff = maxval - minval;
  
    float3 srcRGB = read_imagef(input,sampler,coord).xyz;
    
    int dx = cx - coord.x;
    int dy = cx - coord.y;
    
    if(size.x > size.y){
        dx = (dx * ratio);
    }else{
        dy = (dy * ratio);
    }
    
    int distSq = dx * dx + dy * dy;
    
    float v = ((float)distSq / diff);
    
    float4 dstRGBA = (float4)((srcRGB + v) ,1.0f);
    

    write_imagef(output,coord,dstRGBA);
    
}
