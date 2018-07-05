/* Please Write the OpenCL Kernel(s) code here*/

__kernel void color_overlay_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float4 overlay_color = (float4)(0.0f,0.0f,0.5f,0.5f);
    
    
    float4 src_color = read_imagef(input,sampler,coord);
    
    float4 dst_color; //= mix(src_color,overlay_color,1.0f);
   
    dst_color = (float4)(mix(src_color.xyz / max(src_color.w,0.00390625f),overlay_color.xyz / max(overlay_color.w,0.00390625f),overlay_color.w) *  src_color.w,src_color.w);

   
    write_imagef(output,coord,dst_color);
}