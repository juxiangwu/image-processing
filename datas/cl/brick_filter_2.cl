/* Please Write the OpenCL Kernel(s) code here*/

__kernel void brick_filter_2(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float3 srcRGB = read_imagef(input,sampler,coord).xyz;
    
    float gray = (srcRGB.x + srcRGB.y + srcRGB.z) / 3;
    
    float thresh = 128.0f / 255.0f;
    
    gray = gray >= thresh ? 255 : 0;
    
    float4 dstRGBA = (float4)(gray,gray,gray,1.0f);
    

    write_imagef(output,coord,dstRGBA);
    
}