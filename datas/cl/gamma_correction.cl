/* Please Write the OpenCL Kernel(s) code here*/
__kernel void gamma_correction(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
     float gamma = 1.5f;
     float3 srcRGB = read_imagef(input,sampler,coord).xyz;
     float3 dstRGB = pow(srcRGB,(float3)(1.0f / gamma,1.0f / gamma,1.0f / gamma));
     
     write_imagef(output,coord,(float4)(dstRGB,1.0f));
    
}