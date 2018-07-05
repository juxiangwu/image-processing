/* Please Write the OpenCL Kernel(s) code here*/

__kernel void blur_avg_filter(__read_only image2d_t input, __write_only image2d_t output){
    const int average_mask[9] = {1,1,1,1,1,1,1,1,1};
    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));


    float4 color = (float4)(0,0,0,0);
    int idx = 0;
    for(int i = -1;i <= 1;i++){
        for(int j = -1;j <= 1;j++){
            color += read_imagef(input,sampler,coord + (int2)(i,j)) * average_mask[idx];
            idx++;
        }
    }

   color /= 9.0f;


    write_imagef(output,coord,color);
}