/* Please Write the OpenCL Kernel(s) code here*/
__kernel void bloom_old_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float3 sum = (float3)(0,0,0);
    
    int mask_width = 4;
    int mask_height = 3;
    
    for(int i = -mask_width;i < mask_width;i++){
        for(int j = -mask_height;j < mask_height;j++){
            sum += read_imagef(input,sampler,coord + (int2)(j,i)).xyz * 0.2f;
        }
    }
    
    float4 srcColor= read_imagef(input,sampler,coord);
    float3 dstRGB;
    if(srcColor.x < 0.3f){
        dstRGB = sum * sum * 0.012f + srcColor.xyz;
    }else{
        if(srcColor.x < 0.5f){
            dstRGB = sum * sum * 0.009f + srcColor.xyz;
        }else{
            dstRGB = sum * sum * 0.0075f + srcColor.xyz;
        }
    }
    
    float4 color = (float4)(dstRGB,1.0f);
    
    write_imagef(output,coord,color);
}