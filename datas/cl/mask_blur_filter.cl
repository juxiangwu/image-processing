/* Please Write the OpenCL Kernel(s) code here*/
//not work
__kernel void mask_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
    int radius = 5;
    int d = (radius * 2 + 1);
    int dsize = d * d;
    
    float mask_kernel [] = {
        0.0f,0.0f,1.0f,0.0f,0.0f,
        0.0f,1.0f,3.0f,1.0f,0.0f,
        1.0f,3.0f,7.0f,3.0f,1.0f,
        0.0f,1.0f,3.0f,1.0f,0.0f,
        0.0f,0.0f,1.0f,0.0f,0.0f
    };
   
    const int maskrows = radius / 2;
    const int maskcols = radius / 2;
  
    int idx = 0;
    float3 rgb = (float3)(0.0f,0.0f,0.0f);
    for(int y = -maskrows;y <= maskrows;y++){
        for(int x = -maskcols;x <= maskcols;x++){
            rgb += read_imagef(input,sampler,coord + (int2)(x,y)).xyz * mask_kernel[idx];
            idx++;
        }
    }

    rgb /= dsize;
    write_imagef(output,coord,(float4)(rgb,1.0f));
}