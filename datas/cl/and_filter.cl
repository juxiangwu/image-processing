/* Please Write the OpenCL Kernel(s) code here*/

__kernel void and_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
      
        float3 rgb1 = read_imagef(input,sampler,coord).xyz * 255.0f;
        float3 rgb2 = read_imagef(input2,sampler,coord).xyz * 255.0f;
        
        int3 rgbi1 = convert_int3(rgb1);
        int3 rgbi2 = convert_int3(rgb2);
        
        int3 rgb = (rgbi1 & rgbi2);
      
        write_imagef(output,coord,(float4)(convert_float3(rgb) / 255.0f,1.0f));
    }
}