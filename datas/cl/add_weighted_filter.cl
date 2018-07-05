/* Please Write the OpenCL Kernel(s) code here*/

__kernel void add_weighted_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output,float alpha,float beta,float gamma){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    //float alpha = 0.45f;
    //float beta = 0.55f;
    //float gamma = 2.0f;
    
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
        float3 color = read_imagef(input,sampler,coord).xyz * alpha + read_imagef(input2,sampler,coord).xyz * beta + gamma;
        
        color.xyz = color.xyz / 255.0f;
        write_imagef(output,coord,(float4)(color,1.0f));
    }
}