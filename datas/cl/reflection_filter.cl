/* Please Write the OpenCL Kernel(s) code here*/
__kernel void reflection_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    float4 p_mat[4];
   
    float4 srcRGBA = read_imagef(input,sampler,coord);
    
    srcRGBA *= M_SQRT2_F / length(srcRGBA);
     
    p_mat[0] = (float4)(1.0f,0.0f,0.0f,0.0f) - (srcRGBA * srcRGBA.x);
    p_mat[1] = (float4)(0.0f,1.0f,0.0f,0.0f) - (srcRGBA * srcRGBA.y);
    p_mat[2] = (float4)(0.0f,0.0f,1.0f,0.0f) - (srcRGBA * srcRGBA.z);
    p_mat[3] = (float4)(0.0f,0.0f,0.0f,1.0f) - (srcRGBA * srcRGBA.w);

    float4 dstRGBA;
    
    float4 x_vec = (float4)(0.5f,0.5f,0.5f,1.0f);
    
    dstRGBA.x = dot(p_mat[0],x_vec);
    dstRGBA.y = dot(p_mat[1],x_vec);
    dstRGBA.z = dot(p_mat[2],x_vec);
    dstRGBA.w = dot(p_mat[3],x_vec);

    write_imagef(output,coord,dstRGBA);
    
}