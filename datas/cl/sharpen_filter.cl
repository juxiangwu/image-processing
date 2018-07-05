/* Please Write the OpenCL Kernel(s) code here*/

__kernel void sharpen_filter(__read_only image2d_t input,__write_only image2d_t output){
    const int sharpen_mask [9] = {-0,-2,0,-2,9,-2,0,-2,0};
    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));


    float4 color = (float4)(0,0,0,1);
    int idx = 0;
    for(int i = -1;i <= 1;i++){
        for(int j = -1;j <= 1;j++){
            color += read_imagef(input,sampler,coord + (int2)(i,j)) * sharpen_mask[idx];
            idx++;
        }
    }

   write_imagef(output,coord,color);
}
