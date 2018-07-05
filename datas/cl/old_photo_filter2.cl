/* Please Write the OpenCL Kernel(s) code here*/
__kernel void old_photo_filter(__read_only image2d_t input,__write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;
                               
    const int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float3 color = read_imagef(input,sampler,coord).xyz;
    float3 gray = (color.x + color.y + color.z) / 3.0f;
    float3 dst_color = (float3)(color.x * 0.393f,color.y * 0.349f,color.z * 0.272f);
    dst_color = mix(dst_color,gray,1.0f);
    write_imagef(output,coord,(float4)(dst_color,1.0f));
}