/* Please Write the OpenCL Kernel(s) code here*/

__kernel void prewitt_horizonta_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
    float divider = 3.0f;
    float color_matrix[9] = {
           1/divider, 1/divider, 1/divider,
           0, 0, 0,
           -1/divider, -1/divider, -1/divider
    }; 
    int idx = 0;
    float4 color = (float4)(0,0,0,0);
    for(int i = -1;i <= 1;i++){
        for(int j = -1;j <= 1;j++){
            float4 srcColor = read_imagef(input,sampler,coord + (int2)(j,i));
            color += read_imagef(input,sampler,coord + (int2)(j,i)) * color_matrix[idx];
            idx++;
        }
    }
    
    write_imagef(output,coord,color);
}